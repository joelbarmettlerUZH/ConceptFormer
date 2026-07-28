"""Recursive ConceptFormer: train a graph embedder that composes concept vectors across hops.

Each edge's neighbor is represented by its own (single-token) concept vector rather than its label
embedding, so the encoder COMPOSES over compressed neighbor representations -- a learned graph
embedder, not a text compressor. Iterating the recursion deepens the reach: C1 neighbors -> 2-hop,
C2 neighbors -> 3-hop, ... Supervised by cross-entropy on the MetaQA gold answer (fast
proof-of-concept; the label-free KL version is the paper-grade follow-up).

Knobs that make it iterable and warm-startable:
- ``--neighbor-table``: ``pck1`` computes C1 from a frozen k=1 encoder; a ``.pt`` path loads a
  deeper table (e.g. C2 saved from a 2-hop run) for the next hop.
- ``--init-encoder``: warm-start (continue-train) the main encoder from a checkpoint name or a
  saved ``.pt`` -- from scratch is data-starved at this scale; the 1-hop encoder already maps
  edge features to concepts and only needs to adapt to concept-valued neighbors.
- ``--save-encoder`` / ``--save-table``: persist this level's encoder + its per-entity concept
  table (pooled to one token) so the next hop can continue from them.

Example (2-hop warm-started, then 3-hop continuing from it):
  python scripts/recursive_hop.py --hop 2 --init-encoder pc_k32_best --neighbor-table pck1 \
      --save-encoder e2.pt --save-table c2.pt
  python scripts/recursive_hop.py --hop 3 --init-encoder e2.pt --neighbor-table c2.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from conceptformer.cache import KVCache
from conceptformer.config import settings
from conceptformer.data.metaqa import load_metaqa_qa
from conceptformer.data.snapshot import iter_subgraphs
from conceptformer.generate.signal import answer_ok
from conceptformer.model.backbone import Backbone
from conceptformer.model.chat import ChatModel
from conceptformer.model.featurizer import featurize_subgraph_recursive
from conceptformer.model.injection import pack_embeddings
from conceptformer.train.forcing import gather_path_logits
from conceptformer.train.losses import sequence_cross_entropy
from conceptformer.train.trainer import TEACHER_SYSTEM, ConceptTrainer, TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=int, default=2, choices=(1, 2, 3))
    ap.add_argument("--objective", default="ce", choices=("ce", "kl"),
                    help="ce = supervised gold; kl = label-free distill of teacher-reads-text")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--neighbor-table", default="pck1", help="'pck1' | path to a saved .pt table")
    ap.add_argument("--neighbor-checkpoint", default="pc_k1_best", help="k=1 encoder for 'pck1'")
    ap.add_argument("--init-encoder", default="", help="warm-start: checkpoint name or .pt path")
    ap.add_argument("--save-encoder", default="", help="save trained encoder state_dict here")
    ap.add_argument("--save-table", default="", help="save this level's per-entity concept table")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from conceptformer.cli import _load_trained_checkpoint

    chat = ChatModel(args.model, device=args.device, cache=KVCache(settings.generation_cache_path))
    bb = Backbone(chat)
    main_trainer = ConceptTrainer(
        bb, TrainConfig(k=args.k, d_model=1024, n_layers=4, gate_mode="none",
                        placement="before_entity", total_steps=args.steps, lr=1e-4, seed=0),
    )

    # Warm-start the main encoder (continue-training) from a checkpoint or a saved .pt.
    if args.init_encoder:
        if args.init_encoder.endswith(".pt") and Path(args.init_encoder).exists():
            state = torch.load(args.init_encoder, map_location=args.device)
        else:
            src, _, _ = _load_trained_checkpoint(args.init_encoder, args.model, args.device)
            state = src.model.state_dict()
        missing, unexpected = main_trainer.model.load_state_dict(state, strict=False)
        print(f"warm-started from {args.init_encoder} "
              f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)

    sgs = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / "metaqa")}
    print(f"metaqa 1-hop entities: {len(sgs)}", flush=True)
    hop = args.hop

    # 1-hop = standard label-embedding neighbors (domain fine-tuning on MetaQA, no recursion).
    # 2+ hop = recursive: neighbor half is a concept vector (C1 from a k=1 encoder, or a saved
    # deeper table). Fine-tuning on 1-hop MetaQA first adapts the encoder to the movie domain so
    # the multi-hop runs aren't also paying the Wikidata->MetaQA gap.
    nbr_table: dict | None = None
    if hop >= 2:
        if args.neighbor_table == "pck1":
            print("computing C1 neighbor table from the k=1 encoder…", flush=True)
            nbr_trainer, _, _ = _load_trained_checkpoint(
                args.neighbor_checkpoint, args.model, args.device
            )
            nbr_table = nbr_trainer.precompute_pooled_concepts(sgs.values())
        else:
            print(f"loading neighbor table {args.neighbor_table}…", flush=True)
            nbr_table = torch.load(args.neighbor_table, map_location=args.device)

    def load(path: str, n: int) -> list:
        with Path(path).open(encoding="utf-8") as fh:
            rows = [r for r in load_metaqa_qa(fh) if r.subject_qid in sgs]
        random.Random(0).shuffle(rows)
        return rows[:n]

    base = settings.data_root / "raw/metaqa"
    train = load(str(base / f"qa_train_{hop}hop.txt"), args.n_train)
    test = load(str(base / f"qa_test_{hop}hop.txt"), args.n_eval)
    print(f"train {len(train)} / eval {len(test)} {hop}-hop questions", flush=True)

    from conceptformer.model.featurizer import featurize_subgraph

    feat_cache: dict[str, object] = {}

    def feats(qid: str) -> object:
        if qid not in feat_cache:
            feat_cache[qid] = (
                featurize_subgraph(sgs[qid], bb.embed_labels) if nbr_table is None
                else featurize_subgraph_recursive(sgs[qid], bb.embed_labels, nbr_table)
            )
        return feat_cache[qid]

    def answer_ids(r: object) -> list[int]:
        return bb.tokenizer(r.answer_labels[0], add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def evaluate() -> float:
        main_trainer.model.eval()
        correct = 0
        for i in range(0, len(test), 64):
            chunk = test[i : i + 64]
            preds = main_trainer.generate_student_batch(
                [(sgs[r.subject_qid], r.question) for r in chunk], 32, TEACHER_SYSTEM,
                neighbor_concepts=nbr_table,
            )
            correct += sum(
                answer_ok(pred, row.answer_labels)
                for row, pred in zip(chunk, preds, strict=True)
            )
        return correct / len(test)

    rng = random.Random(1)
    if args.objective == "kl":
        # Label-free KL, the paper objective: distill the frozen teacher READING THE FACTS into
        # the student reading concept tokens. The "target" is the teacher's own greedy path over
        # the answer-guaranteed verbalized neighborhood (no gold label enters the loss), then
        # trainer.prepare + step_prepared run the exact KL step cf-train uses. Hop 1 (label
        # neighbors) only; deeper KL needs recursive features in prepare (future work).
        print("decoding teacher paths (label-free target)…", flush=True)
        tuples = []
        for i in range(0, len(train), 64):
            chunk = train[i : i + 64]
            prompts = [
                (TEACHER_SYSTEM,
                 f"{main_trainer._facts(sgs[r.subject_qid], r.answer_labels[0])}\n\n{r.question}")
                for r in chunk
            ]
            outs = main_trainer._generate_text_batch(prompts, 32)
            for r, o in zip(chunk, outs, strict=True):
                path = bb.tokenizer(o, add_special_tokens=False)["input_ids"]
                if path:
                    tuples.append((sgs[r.subject_qid], r.question, path, r.answer_labels[0]))
        prepared = main_trainer.prepare(tuples)
        print(f"prepared {len(prepared)} KL rows; init {hop}-hop acc: {evaluate():.3f}", flush=True)
        for step in range(1, args.steps + 1):
            main_trainer.model.train()
            loss = main_trainer.step_prepared(rng.sample(prepared, args.batch))
            if step % 200 == 0:
                print(f"step {step}: loss {loss:.3f}", flush=True)
            if step % args.eval_every == 0:
                print(f"step {step}: {hop}-hop acc {evaluate():.3f}", flush=True)
        print(f"FINAL {hop}-hop acc: {evaluate():.3f}", flush=True)
    else:
        print(f"init {hop}-hop acc: {evaluate():.3f}", flush=True)
        for step in range(1, args.steps + 1):
            main_trainer.model.train()
            batch = rng.sample(train, args.batch)
            concepts = main_trainer._encode_concepts([feats(r.subject_qid) for r in batch])
            embeds, ctx, paths, ids_rows = [], [], [], []
            for r in batch:
                path = answer_ids(r)
                head, tail = main_trainer._student_split(
                    TEACHER_SYSTEM, r.question, sgs[r.subject_qid].center.label
                )
                emb, c, idr = main_trainer._student_row(head, tail, path, concepts[len(embeds)])
                embeds.append(emb)
                ctx.append(c)
                paths.append(path)
                ids_rows.append(idr)
            s_in, s_attn = pack_embeddings(embeds)
            s_pos = main_trainer._student_positions(ids_rows, s_attn)
            s_hidden = bb.forward_hidden(s_in, s_attn, s_pos)
            plens = [len(p) for p in paths]
            s_path, mask = gather_path_logits(s_hidden, ctx, plens)
            logits = bb.lm_head(s_path).float()
            targets = main_trainer._pad_targets(paths, mask.shape[1])
            loss = sequence_cross_entropy(logits, targets, mask)
            main_trainer._apply(loss)
            if step % 200 == 0:
                print(f"step {step}: loss {float(loss):.3f}", flush=True)
            if step % args.eval_every == 0:
                print(f"step {step}: {hop}-hop acc {evaluate():.3f}", flush=True)
        print(f"FINAL {hop}-hop acc: {evaluate():.3f}", flush=True)

    if args.save_encoder:
        torch.save(main_trainer.model.state_dict(), args.save_encoder)
        print(f"saved encoder -> {args.save_encoder}", flush=True)
    if args.save_table:  # this level's per-entity concept (pooled to one token) for the next hop
        main_trainer.model.eval()
        table: dict[str, torch.Tensor] = {}
        ents = list(sgs.values())
        with torch.no_grad():
            for i in range(0, len(ents), 64):
                chunk = ents[i : i + 64]
                cs = main_trainer._encode_concepts([feats(sg.center.qid) for sg in chunk])
                for sg, vec in zip(chunk, cs.mean(dim=1), strict=True):
                    table[sg.center.qid] = vec
        torch.save(table, args.save_table)
        print(f"saved concept table ({len(table)}) -> {args.save_table}", flush=True)


if __name__ == "__main__":
    main()

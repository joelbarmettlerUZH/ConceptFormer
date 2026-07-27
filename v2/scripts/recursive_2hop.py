"""Proof-of-concept: train a recursive ConceptFormer that composes 2-hop over concept vectors.

The 1-hop encoder represents a neighbor by its label embedding. Here each neighbor is instead
represented by its own concept vector (from a frozen k=1 encoder), so the main encoder must
COMPOSE over compressed neighbor representations to answer a 2-hop question -- a graph embedder,
not a text compressor. Zero-shot this does not work (the 1-hop-trained encoder never learned to
read concept-valued neighbors); this script TRAINS a fresh k=32 encoder to do it, supervised by
cross-entropy on the MetaQA 2-hop gold answer (a fast proof-of-concept; the label-free KL version
is the paper-grade follow-up).

Run: uv run --group infer python scripts/recursive_2hop.py --steps 4000 --n-train 10000
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
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--neighbor-checkpoint", default="pc_k1_best")
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
    nbr_trainer, _, _ = _load_trained_checkpoint(args.neighbor_checkpoint, args.model, args.device)

    sgs = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / "metaqa")}
    print(f"metaqa 1-hop entities: {len(sgs)}", flush=True)
    print("precomputing k=1 neighbor concept table…", flush=True)
    nbr_table = nbr_trainer.precompute_pooled_concepts(sgs.values())

    def load(path: str, n: int) -> list:
        with Path(path).open(encoding="utf-8") as fh:
            rows = [r for r in load_metaqa_qa(fh) if r.subject_qid in sgs]
        random.Random(0).shuffle(rows)
        return rows[:n]

    train = load(str(settings.data_root / "raw/metaqa/qa_train_2hop.txt"), args.n_train)
    test = load(str(settings.data_root / "raw/metaqa/qa_test_2hop.txt"), args.n_eval)
    print(f"train {len(train)} / eval {len(test)} 2-hop questions", flush=True)

    # Recursive features per unique subject (fixed across steps): [phi(rel); C_1(neighbor)].
    feat_cache: dict[str, object] = {}
    def feats(qid: str) -> object:
        if qid not in feat_cache:
            feat_cache[qid] = featurize_subgraph_recursive(sgs[qid], bb.embed_labels, nbr_table)
        return feat_cache[qid]

    def answer_ids(r: object) -> list[int]:  # teacher-forced target = first gold answer
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

    print(f"init 2-hop acc: {evaluate():.3f}", flush=True)
    rng = random.Random(1)
    for step in range(1, args.steps + 1):
        main_trainer.model.train()
        batch = rng.sample(train, args.batch)
        concepts = main_trainer._encode_concepts([feats(r.subject_qid) for r in batch])
        embeds, ctx, paths, ids_rows = [], [], [], []
        for i, r in enumerate(batch):
            path = answer_ids(r)
            head, tail = main_trainer._student_split(
                TEACHER_SYSTEM, r.question, sgs[r.subject_qid].center.label
            )
            emb, c, idr = main_trainer._student_row(head, tail, path, concepts[i])
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
            print(f"step {step}: 2-hop acc {evaluate():.3f}", flush=True)
    print(f"FINAL 2-hop acc: {evaluate():.3f}", flush=True)


if __name__ == "__main__":
    main()

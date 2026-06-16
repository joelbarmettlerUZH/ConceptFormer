"""ConceptFormer distillation trainer (the integration layer).

Ties together the frozen ``Backbone``, the trainable ``ConceptFormer`` (encoder + gate), the
featurizer, and the KL loss. For each example it runs two forward passes through the *same* frozen
LLM and matches their next-token distributions along the teacher's greedy path:

- **teacher**: ``[system][verbalized facts][question] + path`` (facts as text).
- **student**: ``[system][k concept tokens][question] + path`` (facts as concept tokens).

The concept tokens are spliced where the facts text sat (same slot, matched RoPE positions), via a
placeholder-span split of the chat template (the VLM "image token" trick) so chat special tokens
stay intact. Examples are processed one at a time and the loss is averaged — correctness over speed;
batched/padded forward is a later optimization. See ``docs/MODEL_DESIGN.md`` §0, §5.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from conceptformer.generate.signal import answer_ok
from conceptformer.generate.teacher import TEACHER_SYSTEM
from conceptformer.model.backbone import Backbone
from conceptformer.model.conceptformer import ConceptFormer
from conceptformer.model.featurizer import (
    SubgraphFeatures,
    collate_features,
    featurize_subgraph,
)
from conceptformer.model.injection import build_position_ids, pack_embeddings
from conceptformer.schemas import Subgraph
from conceptformer.train.forcing import gather_path_logits
from conceptformer.train.losses import sequence_cross_entropy, sequence_kl
from conceptformer.verbalize import verbalize_with_answer

# Unique marker reserving the concept-token slot inside the rendered chat template.
_SENTINEL = "\x00CF_CONCEPTS\x00"

# Diverse system prompts to distill under so the concept vectors don't couple to any one prompt
# (prompt augmentation). The vectors must reproduce the teacher under all of these → they encode
# the entity's facts prompt-agnostically. Hold out a *different* prompt at eval to prove decoupling.
AUGMENT_SYSTEMS = (
    "You are a helpful assistant.",
    "You are a helpful assistant. Answer with just the answer, as briefly as possible.",
    "You are a knowledgeable expert. Give accurate, concise answers.",
    "Answer the question using what you know.",
    "Respond helpfully and factually.",
)

# A system prompt deliberately NOT in AUGMENT_SYSTEMS: evaluating under it tests whether the
# concept vectors decoupled from the training prompts (prompt-generalization).
HELD_OUT_EVAL_SYSTEM = "You are a precise question-answering system. State the answer."


def _chunks(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


@dataclass
class TrainConfig:
    k: int = 8
    d_model: int = 512
    n_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.0
    lr: float = 1e-4
    gate_lr: float = 1e-2  # the zero-init gate must open fast or it throttles all learning
    temperature: float = 1.0
    ce_weight: float = 0.0  # optional hard-CE factuality anchor (ablate; 0 = pure KL)
    rag_context_tokens: int = 2048
    warmup_steps: int = 0  # >0 with total_steps enables linear warmup + cosine decay
    total_steps: int = 0
    # Cache the frozen teacher's path hidden states (~m*d/example) so the step skips the teacher
    # forward. A big win for small data x many epochs (high reuse); set False for large-data /
    # few-epoch runs where the teacher is computed ~once anyway and the cache would be huge.
    cache_teacher: bool = True
    # System prompts to distill under (prompt augmentation → prompt-agnostic vectors). Empty =
    # single-prompt (TEACHER_SYSTEM). Set to AUGMENT_SYSTEMS to decouple. Cache scales by the count.
    augment_systems: tuple[str, ...] = ()
    # Neighbor-subsampling: re-sample the teacher's distractor facts each step (answer always kept)
    # so the student must encode the whole neighborhood, not one fixed subset. Forces the live
    # teacher path (no cache) since the teacher target varies per step.
    subsample_neighbors: bool = False
    seed: int = 0


@dataclass
class Prepared:
    """One example with its deterministic preprocessing done once (see ``ConceptTrainer.prepare``).

    Tokenization (teacher context, student tail) and the entity's edge features never change across
    epochs, so we compute them once. ``teacher_hidden`` (``(m, d)``, CPU) caches the *frozen*
    teacher's final hidden states at the path positions — the teacher target is constant, so the
    training step skips the teacher forward entirely and only runs the student. ``qid`` keys the
    per-entity edge-feature cache.
    """

    qid: str
    teacher_ctx_ids: list[int]
    student_tail_ids: list[int]
    path: list[int]
    teacher_hidden: Tensor | None = None
    system_idx: int = 0  # which augmentation system prompt this row was built under


@dataclass
class EvalSet:
    """A self-contained, prepared evaluation set (held-out OR held-in OR any sample).

    Holds everything ``evaluate`` needs so multiple sets can be scored per checkpoint without
    clobbering shared trainer state. ``base``/``teacher`` accuracy is the frozen model's bracket
    on this set (constant across training, so computed once when the set is built).
    """

    items: list
    prepared: list[Prepared]
    base: float
    teacher: float
    system: str = TEACHER_SYSTEM
    max_new: int = 32
    gen_batch: int = 48


class ConceptTrainer:
    """Holds the frozen backbone + trainable ConceptFormer and runs distillation steps."""

    def __init__(self, backbone: Backbone, config: TrainConfig) -> None:
        self.bb = backbone
        self.cfg = config
        d_in = 2 * backbone.d_model  # concat(property, neighbor) edge features
        self.model = ConceptFormer(
            d_in,
            backbone.d_model,
            config.k,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        ).to(backbone.device)
        self.model.train()
        # The gate gets its own (higher) LR: with a shared 1e-4 it stays near 0, which zeroes the
        # gradient to the encoder (grad ∝ tanh(gate)) — a dead zone where nothing learns.
        self.opt = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": config.lr},
                {"params": self.model.gate.parameters(), "lr": config.gate_lr},
            ]
        )
        self.sched = (
            torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_factor)
            if config.total_steps > 0
            else None
        )
        # Deterministic-preprocessing caches (populated by prepare()): per-entity edge features and
        # the constant student "head" token ids (system + user-role header, before the concepts).
        self._feat_cache: dict[str, SubgraphFeatures] = {}
        # System prompts for augmentation (defaults to a single neutral one if not augmenting).
        self._systems = list(config.augment_systems) or [TEACHER_SYSTEM]
        self._head_ids_by_system: dict[int, list[int]] = {}
        self._sub_rng = random.Random(config.seed)  # re-samples distractors when subsampling
        # Default eval set (held-out), populated by setup_eval; brackets static across training.
        self._eval: EvalSet | None = None
        # Cache of frozen base/teacher PopQA brackets per item-set (so a per-checkpoint PopQA
        # trajectory only re-runs the student). Keyed by id(items).
        self._popqa_brackets: dict[int, tuple[float, float]] = {}

    def _lr_factor(self, step: int) -> float:
        """Linear warmup then cosine decay (multiplies each param group's base LR)."""
        warmup, total = self.cfg.warmup_steps, self.cfg.total_steps
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _count_tokens(self, text: str) -> int:
        return len(self.bb.tokenizer.encode(text, add_special_tokens=False))

    def _facts(
        self, sg: Subgraph, answer_qid: str | None = None, rng: random.Random | None = None
    ) -> str:
        # Guarantee the answer's edge is in the teacher's facts; ``rng`` re-samples distractors.
        budget = self.cfg.rag_context_tokens
        return verbalize_with_answer(sg, answer_qid, self._count_tokens, budget, rng)

    def _cached_features(self, sg: Subgraph) -> SubgraphFeatures:
        qid = sg.center.qid
        if qid not in self._feat_cache:
            self._feat_cache[qid] = featurize_subgraph(sg, self.bb.embed_labels)
        return self._feat_cache[qid]

    def _render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        common = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.bb.tokenizer.apply_chat_template(messages, enable_thinking=False, **common)
        except TypeError:
            return self.bb.tokenizer.apply_chat_template(messages, **common)

    def _ids(self, text: str) -> list[int]:
        return self.bb.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _concepts(self, sg: Subgraph) -> Tensor:
        """Encode one subgraph -> gated concept tokens ``(k, d_llm)`` in the backbone dtype."""
        feats = featurize_subgraph(sg, self.bb.embed_labels)
        edge_features = feats.edge_features.unsqueeze(0).float().to(self.bb.device)  # (1, N, 2d)
        mask = torch.ones((1, feats.n_edges), dtype=torch.bool, device=self.bb.device)
        concepts = self.model(edge_features, mask)[0]  # (k, d_llm), fp32, grad
        return concepts.to(self.bb.dtype)

    def _pad_targets(self, paths: list[list[int]], m_max: int) -> Tensor:
        targets = torch.zeros((len(paths), m_max), dtype=torch.long, device=self.bb.device)
        for i, path in enumerate(paths):
            targets[i, : len(path)] = torch.tensor(path, device=self.bb.device)
        return targets

    def _encode_concepts(self, feats: list[SubgraphFeatures]) -> Tensor:
        """One encoder forward over a collated batch of neighborhoods -> gated ``(B, k, d)``."""
        edge_features, mask, _ = collate_features(feats)
        concepts = self.model(edge_features.float().to(self.bb.device), mask.to(self.bb.device))
        return concepts.to(self.bb.dtype)

    def _embed(self, ids: list[int]) -> Tensor:
        return self.bb.embed_tokens(torch.tensor(ids, device=self.bb.device))

    def _forward_kl(
        self,
        teacher_embeds: list[Tensor],
        teacher_ctx: list[int],
        student_embeds: list[Tensor],
        student_ctx: list[int],
        path_lens: list[int],
        paths: list[list[int]],
    ) -> Tensor:
        """One padded forward each (teacher no-grad, student grad), gather paths, KL.

        We forward to hidden states, gather the ~20 path positions, then apply the LM head only
        there — identical logits to a full forward but without the huge ``(B, L, V)`` tensor.
        """
        t_in, t_attn = pack_embeddings(teacher_embeds)
        with torch.no_grad():
            t_hidden = self.bb.forward_hidden(t_in, t_attn, build_position_ids(t_attn))
            t_path, mask_m = gather_path_logits(t_hidden, teacher_ctx, path_lens)  # (B, m, d)
            teacher_g = self.bb.lm_head(t_path)  # (B, m, V)
        s_in, s_attn = pack_embeddings(student_embeds)
        s_hidden = self.bb.forward_hidden(s_in, s_attn, build_position_ids(s_attn))
        s_path, _ = gather_path_logits(s_hidden, student_ctx, path_lens)
        student_g = self.bb.lm_head(s_path).float()
        loss = sequence_kl(student_g, teacher_g.float(), mask_m, temperature=self.cfg.temperature)
        if self.cfg.ce_weight > 0:
            targets = self._pad_targets(paths, mask_m.shape[1])
            loss = loss + self.cfg.ce_weight * sequence_cross_entropy(student_g, targets, mask_m)
        return loss

    def batch_loss(self, batch: list[tuple[Subgraph, str, list[int], str | None]]) -> Tensor:
        """Batched KL recomputing teacher facts each step — the live, no-teacher-cache path.

        Used for neighbor-subsampling (``subsample_neighbors``): facts are re-sampled every step
        (answer guaranteed, distractors shuffled), so the teacher target varies and can't be cached.
        Edge features are still cached (the encoder input is fixed). Also the ``cf-overfit`` path.
        """
        batch = [(sg, q, p, aq) for sg, q, p, aq in batch if p]
        concepts = self._encode_concepts([self._cached_features(sg) for sg, *_ in batch])
        head_emb = self._embed(self._student_head_ids())
        rng = self._sub_rng if self.cfg.subsample_neighbors else None
        teacher_embeds, teacher_ctx, path_lens = [], [], []
        student_embeds, student_ctx, paths = [], [], []
        for i, (sg, question, path, answer_qid) in enumerate(batch):
            facts = self._facts(sg, answer_qid, rng)
            ctx_ids = self._ids(self._render(TEACHER_SYSTEM, f"{facts}\n\n{question}"))
            _, tail_text = self._render(TEACHER_SYSTEM, _SENTINEL + f"\n\n{question}").split(
                _SENTINEL
            )
            tail_ids = self._ids(tail_text)
            teacher_embeds.append(self._embed(ctx_ids + path))
            teacher_ctx.append(len(ctx_ids))
            path_lens.append(len(path))
            paths.append(path)
            student_embeds.append(torch.cat([head_emb, concepts[i], self._embed(tail_ids + path)]))
            student_ctx.append(len(self._student_head_ids()) + self.cfg.k + len(tail_ids))
        return self._forward_kl(
            teacher_embeds, teacher_ctx, student_embeds, student_ctx, path_lens, paths
        )

    def _student_head_ids(self, system_idx: int = 0) -> list[int]:
        """The student prefix (system + user header, before the concepts), cached per system."""
        if system_idx not in self._head_ids_by_system:
            head_text, _ = self._render(self._systems[system_idx], _SENTINEL + "\n\nx").split(
                _SENTINEL
            )
            self._head_ids_by_system[system_idx] = self._ids(head_text)
        return self._head_ids_by_system[system_idx]

    def prepare(self, items: list[tuple[Subgraph, str, list[int], str | None]]) -> list[Prepared]:
        """Deterministic preprocessing once, expanded over the augmentation system prompts.

        Items are ``(subgraph, question, path, answer_qid)``. Each example becomes one ``Prepared``
        per system prompt (diverse prompt distribution); the teacher facts are answer-guaranteed.
        Entity edge features are cached once; the student tail is system-agnostic; the teacher
        context (and cached hidden states) are per-system.
        """
        prepared: list[Prepared] = []
        for sg, question, path, answer_qid in items:
            if not path:
                continue
            qid = sg.center.qid
            if qid not in self._feat_cache:
                self._feat_cache[qid] = featurize_subgraph(sg, self.bb.embed_labels)
            facts = self._facts(sg, answer_qid)
            _, tail_text = self._render(self._systems[0], _SENTINEL + f"\n\n{question}").split(
                _SENTINEL
            )
            tail_ids = self._ids(tail_text)  # system-agnostic (after the user-content start)
            for si, system in enumerate(self._systems):
                ctx_ids = self._ids(self._render(system, f"{facts}\n\n{question}"))
                prepared.append(Prepared(qid, ctx_ids, tail_ids, path, system_idx=si))
        if self.cfg.cache_teacher:
            self._cache_teacher_hidden(prepared)  # one-time frozen-teacher pass (per system)
        return prepared

    @torch.no_grad()
    def _cache_teacher_hidden(self, rows: list[Prepared], chunk_size: int = 32) -> None:
        """Precompute each example's frozen-teacher path-position hidden states (CPU, ~mxd each).

        The teacher target is constant, so we pay one teacher forward here; the step never
        forwards the teacher again — it applies the LM head to these cached hidden states.
        ~320 MB for ~8k examples vs ~48 GB if we cached full logits.
        """
        for chunk in _chunks(rows, chunk_size):
            embeds = [self._embed(r.teacher_ctx_ids + r.path) for r in chunk]
            ctx = [len(r.teacher_ctx_ids) for r in chunk]
            plens = [len(r.path) for r in chunk]
            t_in, t_attn = pack_embeddings(embeds)
            hidden = self.bb.forward_hidden(t_in, t_attn, build_position_ids(t_attn))
            gathered, _ = gather_path_logits(hidden, ctx, plens)  # (B, m_max, d)
            for i, row in enumerate(chunk):
                row.teacher_hidden = gathered[i, : plens[i]].clone().cpu()

    def _padded_teacher(self, batch: list[Prepared]) -> tuple[Tensor, Tensor]:
        """Teacher logits ``(B, m_max, V)`` + path mask from cached hidden states (no forward)."""
        path_lens = [len(p.path) for p in batch]
        m_max = max(path_lens)
        first = batch[0].teacher_hidden
        if first is None:
            raise RuntimeError("teacher_hidden not cached; call prepare() first")
        hidden = first.new_zeros((len(batch), m_max, first.shape[-1]), device=self.bb.device)
        mask = torch.zeros((len(batch), m_max), dtype=torch.bool, device=self.bb.device)
        for i, (p, plen) in enumerate(zip(batch, path_lens, strict=True)):
            h = p.teacher_hidden
            if h is None:
                raise RuntimeError("teacher_hidden not cached; call prepare() first")
            hidden[i, :plen] = h.to(self.bb.device)
            mask[i, :plen] = True
        return self.bb.lm_head(hidden.to(self.bb.dtype)), mask

    def _live_teacher(self, batch: list[Prepared], path_lens: list[int]) -> tuple[Tensor, Tensor]:
        """Teacher logits via a live forward (fallback when the teacher cache is disabled)."""
        teacher_embeds = [self._embed(p.teacher_ctx_ids + p.path) for p in batch]
        teacher_ctx = [len(p.teacher_ctx_ids) for p in batch]
        t_in, t_attn = pack_embeddings(teacher_embeds)
        with torch.no_grad():
            t_hidden = self.bb.forward_hidden(t_in, t_attn, build_position_ids(t_attn))
            t_path, mask_m = gather_path_logits(t_hidden, teacher_ctx, path_lens)
            return self.bb.lm_head(t_path), mask_m

    def batch_loss_cached(self, batch: list[Prepared]) -> Tensor:
        """Batched KL. With the teacher cached, only the student forward runs (~2x faster);
        otherwise the teacher is forwarded live (the large-data / few-epoch path)."""
        concepts = self._encode_concepts([self._feat_cache[p.qid] for p in batch])
        student_embeds, student_ctx, path_lens = [], [], []
        for i, p in enumerate(batch):
            head_ids = self._student_head_ids(p.system_idx)  # per-sampled-system prefix
            head_emb = self._embed(head_ids)
            student_embeds.append(
                torch.cat([head_emb, concepts[i], self._embed(p.student_tail_ids + p.path)])
            )
            student_ctx.append(len(head_ids) + self.cfg.k + len(p.student_tail_ids))
            path_lens.append(len(p.path))

        if batch[0].teacher_hidden is not None:
            teacher_g, mask_m = self._padded_teacher(batch)
        else:
            teacher_g, mask_m = self._live_teacher(batch, path_lens)
        s_in, s_attn = pack_embeddings(student_embeds)
        s_hidden = self.bb.forward_hidden(s_in, s_attn, build_position_ids(s_attn))
        s_path, _ = gather_path_logits(s_hidden, student_ctx, path_lens)
        student_g = self.bb.lm_head(s_path).float()
        loss = sequence_kl(student_g, teacher_g.float(), mask_m, temperature=self.cfg.temperature)
        if self.cfg.ce_weight > 0:
            targets = self._pad_targets([p.path for p in batch], mask_m.shape[1])
            loss = loss + self.cfg.ce_weight * sequence_cross_entropy(student_g, targets, mask_m)
        return loss

    def _apply(self, loss: Tensor) -> float:
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.sched is not None:
            self.sched.step()
        return float(loss.detach())

    def step(self, batch: list[tuple[Subgraph, str, list[int], str | None]]) -> float:
        """One optimizer step from raw ``(subgraph, question, path, answer_qid)`` (live teacher)."""
        return self._apply(self.batch_loss(batch))

    def step_prepared(self, batch: list[Prepared]) -> float:
        """One optimizer step from cached ``Prepared`` rows (GPU-bound)."""
        return self._apply(self.batch_loss_cached(batch))

    def _left_pad_embeds(self, seqs: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Left-pad ``(L_i, d)`` embeds for batched generation (real content right-aligned)."""
        batch, l_max = len(seqs), max(int(s.shape[0]) for s in seqs)
        out = seqs[0].new_zeros((batch, l_max, int(seqs[0].shape[-1])))
        attn = torch.zeros((batch, l_max), dtype=torch.long, device=seqs[0].device)
        for i, s in enumerate(seqs):
            n = int(s.shape[0])
            out[i, l_max - n :] = s
            attn[i, l_max - n :] = 1
        return out, attn

    @torch.no_grad()
    def _generate_text_batch(self, prompts: list[tuple[str, str]], max_new: int) -> list[str]:
        """Batched greedy generation from text prompts (left-padded) — base / RAG-teacher."""
        if not prompts:
            return []
        enc = self.bb.tokenizer(
            [self._render(s, u) for s, u in prompts], return_tensors="pt", padding=True
        ).to(self.bb.device)
        gen = self.bb.model.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            pad_token_id=self.bb.tokenizer.eos_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1] :]
        return [t.strip() for t in self.bb.tokenizer.batch_decode(new, skip_special_tokens=True)]

    @torch.no_grad()
    def generate_student_batch(
        self, items: list[tuple[Subgraph, str]], max_new: int, system: str = TEACHER_SYSTEM
    ) -> list[str]:
        """Batched greedy generation with concept tokens spliced in (left-padded), under ``system``.

        ``system`` may be a prompt the model never trained under — that is the decoupling test.
        """
        if not items:
            return []
        self.model.eval()
        concepts = self._encode_concepts(
            [featurize_subgraph(sg, self.bb.embed_labels) for sg, _ in items]
        )
        head_text, _ = self._render(system, _SENTINEL + "\n\nx").split(_SENTINEL)
        head_emb = self._embed(self._ids(head_text))
        seqs = []
        for i, (_, question) in enumerate(items):
            _, tail = self._render(system, _SENTINEL + f"\n\n{question}").split(_SENTINEL)
            seqs.append(torch.cat([head_emb, concepts[i], self._embed(self._ids(tail))], dim=0))
        in_embeds, attn = self._left_pad_embeds(seqs)
        gen = self.bb.model.generate(
            inputs_embeds=in_embeds, attention_mask=attn, max_new_tokens=max_new,
            do_sample=False, pad_token_id=self.bb.tokenizer.eos_token_id,
        )
        return [t.strip() for t in self.bb.tokenizer.batch_decode(gen, skip_special_tokens=True)]

    def build_eval(
        self,
        val_rows: list,
        sg_by_qid: dict,
        *,
        eval_system: str = TEACHER_SYSTEM,
        max_new: int = 32,
        gen_batch: int = 48,
    ) -> EvalSet:
        """Prepare an eval set ONCE and compute its static base/RAG-teacher accuracy.

        Returns a self-contained ``EvalSet`` (does NOT mutate trainer state), so several sets —
        held-out, held-in, etc. — can be built up front and each scored every checkpoint. Scored
        under ``eval_system``; base and teacher are the frozen model's brackets (constant across
        training), generated once here so per-checkpoint ``evaluate`` only runs the student.
        """
        items = [
            (sg_by_qid[r.subject_qid], r.question, list(r.accepted_answers),
             r.teacher_target_ids, r.answer_qid)
            for r in val_rows
            if r.subject_qid in sg_by_qid and r.teacher_target_ids and r.accepted_answers
        ]
        prepared = self.prepare([(sg, q, p, aq) for sg, q, _, p, aq in items])
        base_ok = teacher_ok = 0
        for chunk in _chunks(items, gen_batch):
            base = self._generate_text_batch([(eval_system, q) for _, q, _, _, _ in chunk], max_new)
            rag_prompts = [
                (eval_system, f"{self._facts(sg, aq)}\n\n{q}") for sg, q, _, _, aq in chunk
            ]
            rag = self._generate_text_batch(rag_prompts, max_new)
            base_ok += sum(answer_ok(p, g) for p, (*_, g, _, _) in zip(base, chunk, strict=True))
            teacher_ok += sum(answer_ok(p, g) for p, (*_, g, _, _) in zip(rag, chunk, strict=True))
        n = max(1, len(items))
        return EvalSet(
            items=items, prepared=prepared, base=base_ok / n, teacher=teacher_ok / n,
            system=eval_system, max_new=max_new, gen_batch=gen_batch,
        )

    def setup_eval(
        self,
        val_rows: list,
        sg_by_qid: dict,
        *,
        eval_system: str = TEACHER_SYSTEM,
        max_new: int = 32,
        gen_batch: int = 48,
    ) -> int:
        """Build the DEFAULT (held-out) eval set and store it. Returns the scorable-row count."""
        self._eval = self.build_eval(
            val_rows, sg_by_qid, eval_system=eval_system, max_new=max_new, gen_batch=gen_batch
        )
        return len(self._eval.items)

    @torch.no_grad()
    def _eval_kl(self, prepared: list[Prepared]) -> float:
        total, n = 0.0, 0
        for chunk in _chunks(prepared, 32):
            total += float(self.batch_loss_cached(chunk)) * len(chunk)
            n += len(chunk)
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, eval_set: EvalSet | None = None) -> dict[str, float]:
        """Per-checkpoint metrics for ``eval_set`` (default: the stored held-out set).

        Student accuracy + KL; base/teacher reused from the set (frozen brackets, computed once).
        """
        es = eval_set if eval_set is not None else self._eval
        if es is None or not es.items:
            return dict.fromkeys(("val_kl", "concept_acc", "base_acc", "teacher_acc", "n_acc"), 0.0)
        self.model.eval()
        concept_ok = 0
        for chunk in _chunks(es.items, es.gen_batch):
            gen_items = [(sg, q) for sg, q, _, _, _ in chunk]
            preds = self.generate_student_batch(gen_items, es.max_new, es.system)
            golds = [it[2] for it in chunk]
            concept_ok += sum(answer_ok(p, g) for p, g in zip(preds, golds, strict=True))
        kl = self._eval_kl(es.prepared)
        self.model.train()
        return {
            "val_kl": kl,
            "concept_acc": concept_ok / len(es.items),
            "base_acc": es.base,
            "teacher_acc": es.teacher,
            "n_acc": len(es.items),
        }

    @torch.no_grad()
    def evaluate_popqa(
        self,
        items: list[tuple[Subgraph, str, list[str], str | None]],
        *,
        eval_system: str = TEACHER_SYSTEM,
        max_new: int = 32,
        gen_batch: int = 48,
        cache_brackets: bool = False,
    ) -> dict[str, float]:
        """External-benchmark accuracy on UNSEEN entities (e.g. PopQA): concept vs base vs RAG.

        ``items`` = ``(subject_subgraph, question, answer_aliases, answer_qid)``, scored under
        ``eval_system`` with the alias matcher (so base/RAG here are this model's own brackets under
        that prompt, not the official PopQA-template numbers). With ``cache_brackets`` the frozen
        base/teacher accuracy (constant across training) is computed once and reused -- so this can
        be called every checkpoint for a trajectory while only the student forward re-runs.
        """
        self.model.eval()
        cached = self._popqa_brackets.get(id(items)) if cache_brackets else None
        concept_ok = base_ok = teacher_ok = 0
        for chunk in _chunks(items, gen_batch):
            concept = self.generate_student_batch(
                [(sg, q) for sg, q, _, _ in chunk], max_new, eval_system
            )
            concept_ok += sum(answer_ok(p, g) for p, (*_, g, _) in zip(concept, chunk, strict=True))
            if cached is None:
                base = self._generate_text_batch(
                    [(eval_system, q) for _, q, _, _ in chunk], max_new
                )
                rag = self._generate_text_batch(
                    [(eval_system, f"{self._facts(sg, aq)}\n\n{q}") for sg, q, _, aq in chunk],
                    max_new,
                )
                base_ok += sum(answer_ok(p, g) for p, (*_, g, _) in zip(base, chunk, strict=True))
                teacher_ok += sum(answer_ok(p, g) for p, (*_, g, _) in zip(rag, chunk, strict=True))
        self.model.train()
        n = max(1, len(items))
        base_acc, teacher_acc = cached if cached is not None else (base_ok / n, teacher_ok / n)
        if cache_brackets and cached is None:
            self._popqa_brackets[id(items)] = (base_acc, teacher_acc)
        return {
            "concept_acc": concept_ok / n,
            "base_acc": base_acc,
            "teacher_acc": teacher_acc,
            "n": len(items),
        }

    def save_checkpoint(self, path: Path) -> None:
        """Persist the trainable ConceptFormer (encoder + gate) + config to reload/eval later."""
        from dataclasses import asdict

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "config": asdict(self.cfg)}, path)

    def gate_values(self) -> list[float]:
        return self.model.gate.gate_values().cpu().tolist()

    def raw_gate(self) -> list[float]:
        """Pre-tanh gate parameters (to see whether the gate is actually opening)."""
        return self.model.gate.gate.detach().cpu().tolist()

    @torch.no_grad()
    def concept_norm(self, sg: Subgraph) -> float:
        """Mean L2 norm of the gated concept tokens for one subgraph (manifold sanity check)."""
        return float(self._concepts(sg).norm(dim=-1).mean())

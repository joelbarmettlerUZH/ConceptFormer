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

import contextlib
import math
import random
from array import array
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from conceptformer.generate.signal import answer_ok
from conceptformer.generate.teacher import TEACHER_SYSTEM
from conceptformer.model.backbone import Backbone
from conceptformer.model.conceptformer import ConceptFormer
from conceptformer.model.featurizer import (
    SubgraphFeatures,
    collate_features,
    featurize_subgraph,
    featurize_subgraph_recursive,
)
from conceptformer.model.injection import ConceptGate, build_position_ids, pack_embeddings
from conceptformer.model.vision_port import (
    VisionPort,
    image_grids,
    mm_token_type_ids,
    vision_block_ids,
)
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


def lr_factor(step: int, warmup: int, total: int, schedule: str) -> float:
    """LR multiplier at ``step``: linear warmup, then ``schedule`` decay (cosine or constant).

    Pure (no trainer state) so the schedule shape is unit-testable without a model. ``constant``
    holds the peak LR after warmup — useful in the few-epoch large-data regime where cosine-to-zero
    over a short horizon decays the LR away before the data is even seen once.
    """
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if schedule == "constant":
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def split_microbatches(items: list, n: int) -> list[list]:
    """Split ``items`` into ``n`` micro-batches for gradient accumulation.

    Strided (``items[i::n]``) rather than contiguous so the groups stay balanced in size even when
    ``len(items)`` is not a multiple of ``n`` -- balanced micro-batches keep the 1/n loss scaling in
    ``ConceptTrainer._apply_accum`` close to a true large-batch gradient.
    """
    return [items[i::n] for i in range(n)]


def place_concept_slot(question: str, entity_label: str | None, placement: str) -> str:
    """Return the user-message content with ``_SENTINEL`` marking the concept-token slot.

    ``prefix`` puts the slot at the message start. The entity-relative modes locate
    ``entity_label`` (case-insensitive) in ``question`` and splice the slot before / after / in
    place of it; if the label isn't present verbatim they fall back to ``prefix``. Pure function
    (no model state) so the placement logic is unit-testable in isolation.
    """
    if placement == "prefix" or not entity_label:
        return _SENTINEL + f"\n\n{question}"
    idx = question.lower().find(entity_label.lower())
    if idx < 0:
        return _SENTINEL + f"\n\n{question}"  # entity not mentioned verbatim -> prefix
    end = idx + len(entity_label)
    before, ent, after = question[:idx], question[idx:end], question[end:]
    if placement == "before_entity":
        q = f"{before}{_SENTINEL} {ent}{after}"
    elif placement == "after_entity":
        q = f"{before}{ent} {_SENTINEL}{after}"
    elif placement == "replace_entity":
        q = f"{before}{_SENTINEL}{after}"
    else:
        raise ValueError(f"unknown placement {placement!r}")
    return f"\n\n{q}"


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
    weight_decay: float = 0.01  # AdamW decoupled wd; 0.01 is AdamW's default (kept for parity)
    rag_context_tokens: int = 2048
    warmup_steps: int = 0  # >0 with total_steps enables linear warmup + the chosen schedule decay
    total_steps: int = 0
    schedule: str = "cosine"  # post-warmup LR decay: "cosine" (to 0) or "constant" (hold peak)
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
    # Max grad-norm (0 = off); guards against a rare bad step committing the run to a poor basin.
    grad_clip: float = 0.0
    # Gradient accumulation: split each optimizer step over this many micro-batches so the EFFECTIVE
    # batch is batch*grad_accum without the memory of a true large batch (batch 64 OOMs on 24 GB).
    # Larger effective batch was the strongest accuracy lever at 72k and it grows with horizon (F8);
    # this reaches batch sizes a single forward can't hold. LR is left UNSCALED (the batch16->32 win
    # used the same LR), keeping batch size a clean separate lever. arXiv:1711.00489 (raise the
    # batch instead of decaying the LR), arXiv:1609.04836 (large-batch generalization gap).
    grad_accum: int = 1
    # Exponential moving average of the trainable weights (0 = off). Eval + the saved checkpoint use
    # the EMA weights, so the final result depends far less on exactly where the last step landed --
    # a direct lever for OUTCOME stability across near-identical runs.
    ema_decay: float = 0.0
    # Concept gate: "tanh" (zero-init, saturating, sign-symmetric -- the suspected init amplifier)
    # or "none" (no gate; the encoder output projection is zero-init instead, so concepts still
    # start at 0 but without the saturating multiplicative coupling).
    gate_mode: str = "tanh"
    # Where the k concept tokens are spliced into the student's user message, relative to the
    # entity mention: "prefix" (message start, the default/baseline), "before_entity",
    # "after_entity", or "replace_entity" (the entity surface form is removed and the concepts
    # stand in for it). Non-prefix modes locate the entity by its label in the question text and
    # fall back to "prefix" if it isn't found verbatim. Teacher-side text is unchanged.
    placement: str = "prefix"
    # Which frozen-model interface the concept tokens enter through: "text" (the embedding
    # stream, default) or "vision" (between <vision_start>/<vision_end> at image-token
    # positions with M-RoPE grid ids, exactly as a k-patch image would — multimodal backbones
    # only; see model/vision_port.py for the faithfulness requirements).
    injection_port: str = "text"


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
    # array('i'), not list[int]: at 915k rows the ~1.1k-token teacher contexts as Python int
    # lists peak past the 124 GB host RAM (the kernel OOM-killed three 100k-corpus launches);
    # 4-byte array storage cuts that term ~9x. The short student/path lists stay lists.
    teacher_ctx_ids: Sequence[int]
    student_head_ids: list[int]  # tokens BEFORE the concept slot (system + any question prefix)
    student_tail_ids: list[int]  # tokens AFTER the concept slot (rest of question + chat close)
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

    def __init__(
        self, backbone: Backbone, config: TrainConfig, concept_model: nn.Module | None = None
    ) -> None:
        self.bb = backbone
        self.cfg = config
        # Seed torch BEFORE building the encoder: its weight init draws from torch's global RNG, so
        # without this every run starts from a different random init -> different optimum -> large
        # run-to-run variance that swamps the effects we measure. (Python's `random`, seeded below,
        # only controlled data order, not weight init / dropout.) CUDA matmul atomics remain a small
        # residual nondeterminism source; this removes the dominant one.
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        d_in = 2 * backbone.d_model  # concat(property, neighbor) edge features
        # ``concept_model`` swaps in an alternative concept producer with the same forward
        # signature — e.g. the untrained TopKMeanEdgeBaseline (eval-only, zero params) — so
        # baselines reuse the exact splice/generate/eval path the trained encoder goes through.
        self.model: nn.Module = (
            concept_model
            if concept_model is not None
            else ConceptFormer(
                d_in,
                backbone.d_model,
                config.k,
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                gate_mode=config.gate_mode,
            )
        ).to(backbone.device)
        self.model.train()
        # The gate gets its own (higher) LR: with a shared 1e-4 it stays near 0, which zeroes the
        # gradient to the encoder (grad ∝ tanh(gate)) — a dead zone where nothing learns.
        groups: list[dict] = []
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            groups.append({"params": encoder.parameters(), "lr": config.lr, "name": "encoder"})
        gate = getattr(self.model, "gate", None)
        if gate is not None:
            groups.append({"params": gate.parameters(), "lr": config.gate_lr, "name": "gate"})
        self.opt = torch.optim.AdamW(groups, weight_decay=config.weight_decay) if groups else None
        self.sched = (
            torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_factor)
            if self.opt is not None and config.total_steps > 0
            else None
        )
        # Deterministic-preprocessing caches (populated by prepare()): per-entity edge features and
        # the constant student "head" token ids (system + user-role header, before the concepts).
        self._feat_cache: dict[str, SubgraphFeatures] = {}
        # System prompts for augmentation (defaults to a single neutral one if not augmenting).
        self._systems = list(config.augment_systems) or [TEACHER_SYSTEM]
        self._sub_rng = random.Random(config.seed)  # re-samples distractors when subsampling
        # EMA of trainable weights (None = off). Eval + checkpoint use these (see _eval_weights).
        self._ema: dict[str, Tensor] | None = None
        if config.ema_decay > 0:
            self._ema = {
                n: p.detach().clone()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }
        # Vision-port splice (multimodal backbones only): the special-token ids delimiting the
        # pseudo-image block the concepts occupy. None = classic text-embedding injection.
        self._vport: VisionPort | None = (
            VisionPort.from_config(backbone.model.config)
            if config.injection_port == "vision"
            else None
        )
        # Vision-patch -> LLM-token merge factor; image_grids needs it so the pseudo-image
        # resolves to exactly k LLM-side tokens (grids are specified in patch units).
        self._vmerge: int = (
            int(getattr(backbone.model.config.vision_config, "spatial_merge_size", 2))
            if self._vport is not None
            else 2
        )
        if config.injection_port not in ("text", "vision"):
            raise ValueError(f"unknown injection_port {config.injection_port!r}")
        # Default eval set (held-out), populated by setup_eval; brackets static across training.
        self._eval: EvalSet | None = None
        self.last_grad_norm = 0.0  # pre-clip total grad-norm of the last step (logged each step)
        # Cache of frozen base/teacher PopQA brackets per item-set (so a per-checkpoint PopQA
        # trajectory only re-runs the student). Keyed by id(items).
        self._popqa_brackets: dict[int, tuple[float, float]] = {}

    @property
    def optimizer(self) -> torch.optim.AdamW:
        """The training optimizer, narrowed; raises for eval-only trainers (zero-param models)."""
        if self.opt is None:
            raise RuntimeError("trainer has no trainable parameters (eval-only concept model)")
        return self.opt

    def _lr_factor(self, step: int) -> float:
        """LR multiplier at ``step`` for this run's schedule (see module-level ``lr_factor``)."""
        return lr_factor(step, self.cfg.warmup_steps, self.cfg.total_steps, self.cfg.schedule)

    def _count_tokens(self, text: str) -> int:
        return len(self.bb.tokenizer.encode(text, add_special_tokens=False))

    def _facts(
        self, sg: Subgraph, answer_qid: str | None = None, rng: random.Random | None = None
    ) -> str:
        # Guarantee the answer's edge is in the teacher's facts; ``rng`` re-samples distractors.
        budget = self.cfg.rag_context_tokens
        return verbalize_with_answer(sg, answer_qid, self._count_tokens, budget, rng)

    def _featurize_cpu(self, sg: Subgraph) -> SubgraphFeatures:
        """Featurize on GPU, cache on CPU.

        The per-entity feature cache must live in host memory: at 100k entities it is
        ~N_edges x 2 d_llm per entity (~11 GB at d=2048), which parked on the GPU starves
        training/eval of headroom and produces flaky OOMs at whatever allocation tips the
        card. The batch paths already move features to the device per step.
        """
        f = featurize_subgraph(sg, self.bb.embed_labels)
        return SubgraphFeatures(edge_features=f.edge_features.cpu(), center=f.center.cpu())

    def _cached_features(self, sg: Subgraph) -> SubgraphFeatures:
        qid = sg.center.qid
        if qid not in self._feat_cache:
            self._feat_cache[qid] = self._featurize_cpu(sg)
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

    def _student_row(
        self, head_ids: list[int], tail_ids: list[int], path: list[int], concept: Tensor
    ) -> tuple[Tensor, int, list[int]]:
        """One student sequence -> (embeddings, CONTEXT length, full ids incl. the path).

        The context length excludes ``path`` (teacher-forced target tokens appended after the
        prompt) — ``gather_path_logits`` indexes relative to it. Text port: ``[head][concepts]
        [tail][path]``; the returned ids are empty (text positions come from the attention mask
        alone). Vision port: the concepts sit at image-token positions inside a
        ``[vision_start]..[vision_end]`` block; the ids are REAL and cover the whole row because
        M-RoPE derives every position, path included, from them (``_student_positions``).
        """
        k = self.cfg.k
        if self._vport is None:
            emb = torch.cat([self._embed(head_ids), concept, self._embed(tail_ids + path)])
            return emb, len(head_ids) + k + len(tail_ids), []
        vp = self._vport
        emb = torch.cat(
            [
                self._embed([*head_ids, vp.vision_start_id]),
                concept,
                self._embed([vp.vision_end_id, *tail_ids, *path]),
            ]
        )
        ids = head_ids + vision_block_ids(vp, k) + tail_ids + path
        return emb, len(head_ids) + k + 2 + len(tail_ids), ids

    def _student_positions(self, ids_rows: list[list[int]], attn: Tensor) -> Tensor:
        """Packed-batch position ids: mask-derived (text port) or M-RoPE grids (vision port)."""
        if self._vport is None:
            return build_position_ids(attn)
        pad = int(self.bb.tokenizer.eos_token_id or 0)  # pad positions are attention-masked
        ids = torch.full(
            (len(ids_rows), int(attn.shape[1])), pad, dtype=torch.long, device=self.bb.device
        )
        for i, row in enumerate(ids_rows):
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.bb.device)
        pos, _ = self.bb.mrope_position_ids(
            ids,
            mm_token_type_ids(ids, self._vport),
            image_grids(len(ids_rows), self.cfg.k, self.bb.device, merge=self._vmerge),
            attn,
        )
        return pos

    def _forward_kl(
        self,
        teacher_embeds: list[Tensor],
        teacher_ctx: list[int],
        student_embeds: list[Tensor],
        student_ctx: list[int],
        path_lens: list[int],
        paths: list[list[int]],
        student_ids: list[list[int]],
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
        s_pos = self._student_positions(student_ids, s_attn)
        s_hidden = self.bb.forward_hidden(s_in, s_attn, s_pos)
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
        rng = self._sub_rng if self.cfg.subsample_neighbors else None
        teacher_embeds, teacher_ctx, path_lens = [], [], []
        student_embeds, student_ctx, paths = [], [], []
        student_ids: list[list[int]] = []
        for i, (sg, question, path, answer_qid) in enumerate(batch):
            facts = self._facts(sg, answer_qid, rng)
            ctx_ids = self._ids(self._render(TEACHER_SYSTEM, f"{facts}\n\n{question}"))
            label = sg.center.label or sg.center.qid
            head_ids, tail_ids = self._student_split(TEACHER_SYSTEM, question, label)
            teacher_embeds.append(self._embed(ctx_ids + path))
            teacher_ctx.append(len(ctx_ids))
            path_lens.append(len(path))
            paths.append(path)
            emb, ctx, ids = self._student_row(head_ids, tail_ids, path, concepts[i])
            student_embeds.append(emb)
            student_ctx.append(ctx)
            student_ids.append(ids)
        return self._forward_kl(
            teacher_embeds, teacher_ctx, student_embeds, student_ctx, path_lens, paths,
            student_ids,
        )

    def _student_split(
        self, system: str, question: str, entity_label: str | None
    ) -> tuple[list[int], list[int]]:
        """Token ids (before, after) the concept slot for this (system, question, placement)."""
        slot = place_concept_slot(question, entity_label, self.cfg.placement)
        head_text, tail_text = self._render(system, slot).split(_SENTINEL)
        return self._ids(head_text), self._ids(tail_text)

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
                self._feat_cache[qid] = self._featurize_cpu(sg)
            facts = self._facts(sg, answer_qid)
            label = sg.center.label or sg.center.qid
            for si, system in enumerate(self._systems):
                ctx_ids = array("i", self._ids(self._render(system, f"{facts}\n\n{question}")))
                head_ids, tail_ids = self._student_split(system, question, label)
                prepared.append(Prepared(qid, ctx_ids, head_ids, tail_ids, path, system_idx=si))
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
            embeds = [self._embed([*r.teacher_ctx_ids, *r.path]) for r in chunk]
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
        teacher_embeds = [self._embed([*p.teacher_ctx_ids, *p.path]) for p in batch]
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
        student_ids: list[list[int]] = []
        for i, p in enumerate(batch):
            emb, ctx, ids = self._student_row(
                p.student_head_ids, p.student_tail_ids, p.path, concepts[i]
            )
            student_embeds.append(emb)
            student_ctx.append(ctx)
            student_ids.append(ids)
            path_lens.append(len(p.path))

        if batch[0].teacher_hidden is not None:
            teacher_g, mask_m = self._padded_teacher(batch)
        else:
            teacher_g, mask_m = self._live_teacher(batch, path_lens)
        s_in, s_attn = pack_embeddings(student_embeds)
        s_pos = self._student_positions(student_ids, s_attn)
        s_hidden = self.bb.forward_hidden(s_in, s_attn, s_pos)
        s_path, _ = gather_path_logits(s_hidden, student_ctx, path_lens)
        student_g = self.bb.lm_head(s_path).float()
        loss = sequence_kl(student_g, teacher_g.float(), mask_m, temperature=self.cfg.temperature)
        if self.cfg.ce_weight > 0:
            targets = self._pad_targets([p.path for p in batch], mask_m.shape[1])
            loss = loss + self.cfg.ce_weight * sequence_cross_entropy(student_g, targets, mask_m)
        return loss

    def _optimizer_step(self) -> None:
        """Clip (recording pre-clip norm), step, schedule, EMA — shared by single + accumulated."""
        # Always record the pre-clip total grad-norm (a key instability signal); clip if enabled.
        clip = self.cfg.grad_clip if self.cfg.grad_clip > 0 else float("inf")
        self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip))
        self.optimizer.step()
        if self.sched is not None:
            self.sched.step()
        if self._ema is not None:
            d = self.cfg.ema_decay
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in self._ema:
                        self._ema[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def _apply(self, loss: Tensor) -> float:
        self.optimizer.zero_grad()
        loss.backward()
        self._optimizer_step()
        return float(loss.detach())

    def _apply_accum(
        self, loss_fn: Callable[[list[Any]], Tensor], micro_batches: list[list[Any]]
    ) -> float:
        """One optimizer step accumulated over several micro-batches (effective batch = their sum).

        Each micro loss is scaled by 1/n and the grads summed. Because ``sequence_kl`` averages over
        supervised *tokens* (not examples), this equals a true large-batch gradient only when the
        micro-batches carry equal token counts; with variable path lengths it is the standard
        accumulation approximation (each micro-batch weighted equally, not by token count) -- the
        same trade-off every framework's grad-accum makes. Losses are computed and freed one
        micro-batch at a time, so peak memory stays at one batch (batch 64 OOMs as one forward).
        """
        self.optimizer.zero_grad()
        n = len(micro_batches)
        total = 0.0
        for mb in micro_batches:
            loss = loss_fn(mb) / n
            loss.backward()
            total += float(loss.detach())
        self._optimizer_step()
        return total

    @contextlib.contextmanager
    def _eval_weights(self) -> Iterator[None]:
        """Swap EMA weights into the model for eval/checkpoint, then restore (no-op if EMA off)."""
        if self._ema is None:
            yield
            return
        backup = {n: p.detach().clone() for n, p in self.model.named_parameters() if n in self._ema}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self._ema:
                    p.copy_(self._ema[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])

    def step(self, batch: list[tuple[Subgraph, str, list[int], str | None]]) -> float:
        """One optimizer step from raw ``(subgraph, question, path, answer_qid)`` (live teacher)."""
        return self._apply(self.batch_loss(batch))

    def step_prepared(self, batch: list[Prepared]) -> float:
        """One optimizer step from cached ``Prepared`` rows (GPU-bound)."""
        return self._apply(self.batch_loss_cached(batch))

    def step_accum(
        self, micro_batches: list[list[tuple[Subgraph, str, list[int], str | None]]]
    ) -> float:
        """Accumulated step from raw 4-tuple micro-batches (live teacher; subsample path)."""
        return self._apply_accum(self.batch_loss, micro_batches)

    def step_prepared_accum(self, micro_batches: list[list[Prepared]]) -> float:
        """Accumulated step from cached ``Prepared`` micro-batches (the cached-teacher path)."""
        return self._apply_accum(self.batch_loss_cached, micro_batches)

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
    @torch.no_grad()
    def precompute_pooled_concepts(
        self, subgraphs: Sequence[Subgraph], batch_size: int = 64
    ) -> dict[str, Tensor]:
        """{qid: mean-pooled concept vector ``(d,)``} for every subgraph -- the 2-hop probe table.

        Encodes each entity's 1-hop neighborhood once and averages the ``k`` concept tokens into a
        single ``d`` vector, so it can slot into a neighbor feature slot (see
        ``featurize_subgraph_recursive``). Kept on the model device/dtype for a cheap cat later.
        """
        self.model.eval()
        items = list(subgraphs)
        table: dict[str, Tensor] = {}
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            feats = [featurize_subgraph(sg, self.bb.embed_labels) for sg in chunk]
            pooled = self._encode_concepts(feats).mean(dim=1)  # (B, d)
            for sg, vec in zip(chunk, pooled, strict=True):
                table[sg.center.qid] = vec
        return table

    def generate_student_batch(
        self, items: list[tuple[Subgraph, str]], max_new: int, system: str = TEACHER_SYSTEM,
        neighbor_concepts: dict[str, Tensor] | None = None,
    ) -> list[str]:
        """Batched greedy generation with concept tokens spliced in (left-padded), under ``system``.

        ``system`` may be a prompt the model never trained under — that is the decoupling test.
        If ``neighbor_concepts`` is given, each edge's neighbor is featurized by its pooled concept
        vector instead of its label (the recursive 2-hop probe).
        """
        if not items:
            return []
        self.model.eval()
        if neighbor_concepts is None:
            feats = [featurize_subgraph(sg, self.bb.embed_labels) for sg, _ in items]
        else:
            feats = [
                featurize_subgraph_recursive(sg, self.bb.embed_labels, neighbor_concepts)
                for sg, _ in items
            ]
        concepts = self._encode_concepts(feats)
        seqs, ids_rows = [], []
        for i, (sg, question) in enumerate(items):
            label = sg.center.label or sg.center.qid
            head_ids, tail_ids = self._student_split(system, question, label)
            emb, _ctx, ids = self._student_row(head_ids, tail_ids, [], concepts[i])
            seqs.append(emb)
            ids_rows.append(ids)
        if self._vport is not None:
            return self._generate_vision_batch(seqs, ids_rows, max_new)
        in_embeds, attn = self._left_pad_embeds(seqs)
        gen = self.bb.model.generate(
            inputs_embeds=in_embeds, attention_mask=attn, max_new_tokens=max_new,
            do_sample=False, pad_token_id=self.bb.tokenizer.eos_token_id,
        )
        return [t.strip() for t in self.bb.tokenizer.batch_decode(gen, skip_special_tokens=True)]

    @torch.no_grad()
    def _generate_vision_batch(
        self, seqs: list[Tensor], ids_rows: list[list[int]], max_new: int
    ) -> list[str]:
        """Greedy decode for the vision port via a manual KV-cached loop.

        The wrapper's ``generate`` derives M-RoPE from token ids, which spliced
        ``inputs_embeds`` cannot carry — so we compute grid positions for the prefix with
        ``mrope_position_ids`` and continue generated tokens at ``seq_len + step + delta``.
        """
        vp = self._vport
        if vp is None:
            raise RuntimeError("vision generation requires injection_port='vision'")
        embeds, attn = self._left_pad_embeds(seqs)
        batch, l_max = attn.shape
        eos = int(self.bb.tokenizer.eos_token_id)
        ids = torch.full((batch, l_max), eos, dtype=torch.long, device=self.bb.device)
        for i, row in enumerate(ids_rows):
            ids[i, l_max - len(row) :] = torch.tensor(
                row, dtype=torch.long, device=self.bb.device
            )
        pos, deltas = self.bb.mrope_position_ids(
            ids,
            mm_token_type_ids(ids, vp),
            image_grids(batch, self.cfg.k, self.bb.device, merge=self._vmerge),
            attn,
        )
        out = self.bb.forward_cached(embeds.to(self.bb.dtype), attn, pos)
        next_logits = self.bb.lm_head(out.last_hidden_state[:, -1])
        past = out.past_key_values
        deltas = deltas.view(-1).to(self.bb.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=self.bb.device)
        collected: list[list[int]] = [[] for _ in range(batch)]
        for step in range(max_new):
            next_id = next_logits.argmax(dim=-1)  # (B,)
            next_id = torch.where(finished, torch.full_like(next_id, eos), next_id)
            for i, token in enumerate(next_id.tolist()):
                if not bool(finished[i]):
                    collected[i].append(int(token))
            finished |= next_id == eos
            if bool(finished.all()) or step == max_new - 1:
                break
            step_emb = self.bb.embed_tokens(next_id).unsqueeze(1)  # (B, 1, d)
            attn = torch.cat(
                [attn, torch.ones((batch, 1), dtype=attn.dtype, device=attn.device)], dim=1
            )
            # Generated tokens continue TEXT positions: all three M-RoPE planes equal, at
            # seq_len + step + delta (delta = the grid's compression of absolute positions).
            pos_step = (l_max + step + deltas).view(1, batch, 1).expand(3, batch, 1)
            out = self.bb.forward_cached(step_emb, attn, pos_step, past_key_values=past)
            past = out.past_key_values
            next_logits = self.bb.lm_head(out.last_hidden_state[:, -1])
        texts = []
        for row_ids in collected:
            if eos in row_ids:
                row_ids = row_ids[: row_ids.index(eos)]
            texts.append(self.bb.tokenizer.decode(row_ids, skip_special_tokens=True).strip())
        return texts

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
        # Chunk of 8, not the training batch: in the live-teacher regime this path runs a FULL
        # teacher forward over long facts-sequences plus two (B, m, V) float32 logit tensors —
        # at backbones >=1.7B a chunk of 32 OOMs a 24 GB card at the very first eval.
        total, n = 0.0, 0
        for chunk in _chunks(prepared, 8):
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
        with self._eval_weights():  # eval under EMA weights when enabled
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
        with self._eval_weights():  # student generation under EMA weights when enabled
            for chunk in _chunks(items, gen_batch):
                concept = self.generate_student_batch(
                    [(sg, q) for sg, q, _, _ in chunk], max_new, eval_system
                )
                concept_ok += sum(
                    answer_ok(p, g) for p, (*_, g, _) in zip(concept, chunk, strict=True)
                )
                if cached is None:
                    base = self._generate_text_batch(
                        [(eval_system, q) for _, q, _, _ in chunk], max_new
                    )
                    rag = self._generate_text_batch(
                        [(eval_system, f"{self._facts(sg, aq)}\n\n{q}") for sg, q, _, aq in chunk],
                        max_new,
                    )
                    base_ok += sum(
                        answer_ok(p, g) for p, (*_, g, _) in zip(base, chunk, strict=True)
                    )
                    teacher_ok += sum(
                        answer_ok(p, g) for p, (*_, g, _) in zip(rag, chunk, strict=True)
                    )
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

    def save_checkpoint(self, path: Path, meta: dict | None = None) -> None:
        """Persist the trainable ConceptFormer + config (the EMA weights when EMA is on).

        ``meta`` records run facts the TrainConfig doesn't carry but re-evaluation must
        reconstruct exactly — the split mode/seed/fraction (eval-final rebuilds the checkpoint's
        train/val split from it; absent = legacy question-level split with the training seed).
        """
        from dataclasses import asdict

        path.parent.mkdir(parents=True, exist_ok=True)
        with self._eval_weights():
            state = {n: p.detach().cpu().clone() for n, p in self.model.state_dict().items()}
        blob: dict = {"model": state, "config": asdict(self.cfg)}
        if meta:
            blob["meta"] = meta
        torch.save(blob, path)

    def gate_values(self) -> list[float]:
        gate = getattr(self.model, "gate", None)
        return gate.gate_values().cpu().tolist() if isinstance(gate, ConceptGate) else []

    def raw_gate(self) -> list[float]:
        """Pre-tanh gate parameters (to see whether the gate is actually opening)."""
        gate = getattr(self.model, "gate", None)
        return gate.gate.detach().cpu().tolist() if isinstance(gate, ConceptGate) else []

    @torch.no_grad()
    def concept_norm(self, sg: Subgraph) -> float:
        """Mean L2 norm of the gated concept tokens for one subgraph (manifold sanity check)."""
        return float(self._concepts(sg).norm(dim=-1).mean())

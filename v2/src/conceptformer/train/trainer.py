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

from dataclasses import dataclass

import torch
from torch import Tensor

from conceptformer.generate.teacher import TEACHER_SYSTEM
from conceptformer.model.backbone import Backbone
from conceptformer.model.conceptformer import ConceptFormer
from conceptformer.model.featurizer import featurize_subgraph
from conceptformer.schemas import Subgraph
from conceptformer.train.forcing import gather_path_logits
from conceptformer.train.losses import sequence_cross_entropy, sequence_kl
from conceptformer.verbalize import verbalize_budgeted

# Unique marker reserving the concept-token slot inside the rendered chat template.
_SENTINEL = "\x00CF_CONCEPTS\x00"


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

    def _count_tokens(self, text: str) -> int:
        return len(self.bb.tokenizer.encode(text, add_special_tokens=False))

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

    def _teacher_path_logits(
        self, sg: Subgraph, question: str, path_ids: list[int]
    ) -> Tensor:
        """Frozen teacher logits aligned to the path: facts-as-text in context."""
        facts = verbalize_budgeted(sg, self._count_tokens, self.cfg.rag_context_tokens)
        context_ids = self._ids(self._render(TEACHER_SYSTEM, f"{facts}\n\n{question}"))
        seq = torch.tensor(context_ids + path_ids, device=self.bb.device)
        embeds = self.bb.embed_tokens(seq).unsqueeze(0)
        attn = torch.ones((1, embeds.shape[1]), dtype=torch.long, device=self.bb.device)
        with torch.no_grad():
            logits = self.bb.forward_embeds(embeds, attn)
        gathered, _ = gather_path_logits(logits, [len(context_ids)], [len(path_ids)])
        return gathered  # (1, m, V)

    def _student_path_logits(
        self, sg: Subgraph, question: str, path_ids: list[int]
    ) -> tuple[Tensor, Tensor]:
        """Student logits aligned to the path: concept tokens spliced into the facts slot."""
        head_text, tail_text = self._render(TEACHER_SYSTEM, _SENTINEL + f"\n\n{question}").split(
            _SENTINEL
        )
        head_ids, tail_ids = self._ids(head_text), self._ids(tail_text)
        concepts = self._concepts(sg)  # (k, d), grad
        head = self.bb.embed_tokens(torch.tensor(head_ids, device=self.bb.device))
        tail = self.bb.embed_tokens(torch.tensor(tail_ids + path_ids, device=self.bb.device))
        embeds = torch.cat([head, concepts, tail], dim=0).unsqueeze(0)  # (1, L, d)
        attn = torch.ones((1, embeds.shape[1]), dtype=torch.long, device=self.bb.device)
        logits = self.bb.forward_embeds(embeds, attn)
        context_len = len(head_ids) + self.cfg.k + len(tail_ids)
        gathered, mask = gather_path_logits(logits, [context_len], [len(path_ids)])
        return gathered, mask  # (1, m, V), (1, m)

    def example_loss(self, sg: Subgraph, question: str, path_ids: list[int]) -> Tensor:
        teacher = self._teacher_path_logits(sg, question, path_ids).float()
        student, mask = self._student_path_logits(sg, question, path_ids)
        student = student.float()
        loss = sequence_kl(student, teacher, mask, temperature=self.cfg.temperature)
        if self.cfg.ce_weight > 0:
            target = torch.tensor([path_ids], device=self.bb.device)
            loss = loss + self.cfg.ce_weight * sequence_cross_entropy(student, target, mask)
        return loss

    def step(self, batch: list[tuple[Subgraph, str, list[int]]]) -> float:
        """One optimizer step over a list of ``(subgraph, question, teacher_path_ids)``."""
        self.opt.zero_grad()
        losses = [self.example_loss(sg, q, p) for sg, q, p in batch if p]
        loss = torch.stack(losses).mean()
        loss.backward()
        self.opt.step()
        return float(loss.detach())

    def gate_values(self) -> list[float]:
        return self.model.gate.gate_values().cpu().tolist()

    def raw_gate(self) -> list[float]:
        """Pre-tanh gate parameters (to see whether the gate is actually opening)."""
        return self.model.gate.gate.detach().cpu().tolist()

    @torch.no_grad()
    def concept_norm(self, sg: Subgraph) -> float:
        """Mean L2 norm of the gated concept tokens for one subgraph (manifold sanity check)."""
        return float(self._concepts(sg).norm(dim=-1).mean())

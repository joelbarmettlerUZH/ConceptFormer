"""Offline tests for the multimodal-wrapper Backbone adapter (Qwen3.5-style nesting)."""

from types import SimpleNamespace

import torch
from torch import nn

from conceptformer.model.backbone import Backbone


class _Stack(nn.Module):
    """Records the position_ids it was called with; returns embeddings as hidden states."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_position_ids: object = "unset"

    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None):
        self.seen_position_ids = position_ids
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def _chat(multimodal: bool):
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(10, 4)
            self.lm_head = nn.Linear(4, 10, bias=False)
            stack = _Stack()
            self.model = (
                SimpleNamespace(language_model=stack, visual=object()) if multimodal else stack
            )

        def get_input_embeddings(self) -> nn.Embedding:
            return self.emb

        def eval(self):  # SimpleNamespace member breaks nn.Module.eval recursion
            return self

        def requires_grad_(self, requires_grad: bool = True):
            return self

    return SimpleNamespace(model=Model(), tokenizer=None, _device="cpu")


def test_multimodal_wrapper_routes_to_language_model_and_drops_position_ids():
    bb = Backbone(_chat(multimodal=True))
    assert bb.is_multimodal
    x = torch.zeros((1, 3, 4))
    attn = torch.ones((1, 3), dtype=torch.long)
    out = bb.forward_hidden(x, attn, torch.arange(3).unsqueeze(0))
    stack = bb._lm
    # M-RoPE stacks expect (3, B, L) ids; the adapter passes None so the stack derives them.
    assert stack.seen_position_ids is None
    assert out.shape == (1, 3, 4)


def test_plain_causal_model_keeps_position_ids():
    bb = Backbone(_chat(multimodal=False))
    assert not bb.is_multimodal
    x = torch.zeros((1, 3, 4))
    attn = torch.ones((1, 3), dtype=torch.long)
    pos = torch.arange(3).unsqueeze(0)
    bb.forward_hidden(x, attn, pos)
    assert torch.equal(bb._lm.seen_position_ids, pos)

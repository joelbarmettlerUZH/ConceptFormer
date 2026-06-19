"""Gradient-accumulation correctness for ``ConceptTrainer._apply_accum``.

These run without a backbone: ``_apply_accum`` only touches ``opt``/``model``/``sched``/``_ema``/
``cfg.grad_clip``, so a trainer built via ``__new__`` over a tiny ``nn.Linear`` is enough to lock
the maths -- the part that would silently break (wrong loss scaling = wrong LR).
"""

from __future__ import annotations

import torch
from torch import nn

from conceptformer.train.trainer import ConceptTrainer, split_microbatches


def test_split_microbatches_covers_every_item_once_and_balanced() -> None:
    items = list(range(10))
    micro = split_microbatches(items, 3)
    assert len(micro) == 3
    assert sorted(x for mb in micro for x in mb) == items  # partition: no drops, no dupes
    assert max(len(mb) for mb in micro) - min(len(mb) for mb in micro) <= 1  # balanced


class _Cfg:
    grad_clip = 0.0


def _bare_trainer(model: nn.Module, lr: float) -> ConceptTrainer:
    # Bypass __init__ (which loads the LLM backbone); wire only what _apply_accum reads.
    t = ConceptTrainer.__new__(ConceptTrainer)
    t.model = model
    t.opt = torch.optim.SGD(model.parameters(), lr=lr)
    t.sched = None
    t._ema = None
    t.last_grad_norm = 0.0
    t.cfg = _Cfg()  # type: ignore[assignment]
    return t


def test_accum_equals_single_large_batch_step() -> None:
    """Accumulating n micro-batches (1/n scaled) must match one step over the concatenated batch.

    Equal-sized micro-batches + a mean-reduction loss => the token-weighting approximation noted in
    ``_apply_accum`` is exact, so the parameter update must match to floating-point tolerance. A
    missing 1/n scale (the likely bug) would make the accumulated step n x too large and fail here.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    init_state = {k: v.clone() for k, v in nn.Linear(4, 1).state_dict().items()}

    def loss_fn(model: nn.Module, batch: list[int]) -> torch.Tensor:
        idx = torch.tensor(batch)
        return nn.functional.mse_loss(model(x[idx]), y[idx])  # mean reduction

    ref_model = nn.Linear(4, 1)
    ref_model.load_state_dict(init_state)
    ref = _bare_trainer(ref_model, lr=0.1)
    ref._apply(loss_fn(ref_model, list(range(8))))
    expected = [p.detach().clone() for p in ref_model.parameters()]

    acc_model = nn.Linear(4, 1)
    acc_model.load_state_dict(init_state)
    acc = _bare_trainer(acc_model, lr=0.1)
    acc._apply_accum(lambda mb: loss_fn(acc_model, mb), split_microbatches(list(range(8)), 2))

    for got, exp in zip(acc_model.parameters(), expected, strict=True):
        assert torch.allclose(got, exp, atol=1e-6)

"""Offline CPU tests for the ConceptFormer encoder (tiny dims; the structural invariants)."""

import pytest

torch = pytest.importorskip("torch")  # encoder needs the `infer` group; skip cleanly if absent

from conceptformer.model.encoder import ConceptEncoder  # noqa: E402


def _encoder(k=4, d_in=16, d_llm=24, d_model=32) -> ConceptEncoder:
    torch.manual_seed(0)
    enc = ConceptEncoder(d_in=d_in, d_llm=d_llm, k=k, d_model=d_model, n_layers=2, n_heads=4)
    return enc.eval()  # eval → no dropout → deterministic


def test_output_shape_is_k_by_dllm_regardless_of_N():
    enc = _encoder(k=4, d_in=16, d_llm=24)
    for n in (1, 3, 50):  # k output tokens independent of neighbor count N
        feats = torch.randn(2, n, 16)
        mask = torch.ones(2, n, dtype=torch.bool)
        out = enc(feats, mask)
        assert out.shape == (2, 4, 24)


def test_permutation_invariance_over_neighbors():
    enc = _encoder()
    feats = torch.randn(1, 6, 16)
    mask = torch.ones(1, 6, dtype=torch.bool)
    perm = torch.randperm(6)
    out = enc(feats, mask)
    out_perm = enc(feats[:, perm, :], mask[:, perm])
    # neighbors are an unordered set → reordering them must not change the concept tokens
    assert torch.allclose(out, out_perm, atol=1e-5)


def test_padded_neighbors_do_not_affect_output():
    enc = _encoder()
    feats = torch.randn(1, 4, 16)
    mask = torch.tensor([[True, True, True, False]])  # last neighbor is padding
    out = enc(feats, mask)
    # arbitrarily corrupting the masked-out neighbor's features must not change the output
    corrupted = feats.clone()
    corrupted[0, 3, :] = torch.randn(16) * 100
    out2 = enc(corrupted, mask)
    assert torch.allclose(out, out2, atol=1e-6)


def test_different_valid_neighbors_change_output():
    enc = _encoder()
    feats = torch.randn(1, 3, 16)
    mask = torch.ones(1, 3, dtype=torch.bool)
    out = enc(feats, mask)
    changed = feats.clone()
    changed[0, 1, :] += 5.0  # a *real* neighbor changed → output must move
    assert not torch.allclose(out, enc(changed, mask), atol=1e-4)


def test_latent_init_seeds_queries():
    init = torch.randn(4, 32)
    enc = ConceptEncoder(d_in=16, d_llm=24, k=4, d_model=32, latent_init=init)
    assert torch.allclose(enc.latents.detach(), init)
    with pytest.raises(ValueError, match="latent_init must be"):
        ConceptEncoder(d_in=16, d_llm=24, k=4, d_model=32, latent_init=torch.randn(3, 32))


def test_gradients_flow_to_encoder_params():
    enc = _encoder()
    feats = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    enc(feats, mask).pow(2).mean().backward()
    assert enc.latents.grad is not None and enc.latents.grad.abs().sum() > 0
    assert enc.out_proj.weight.grad is not None

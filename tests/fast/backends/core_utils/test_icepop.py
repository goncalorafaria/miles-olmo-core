"""Numerical IcePop contracts and dispatch through the real MILES loss adapter."""

import numpy as np
import pytest
import torch
from tests.fast.backends.training_utils.test_score_centering_pipeline import single_rank  # noqa: F401
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.backends.core_utils.icepop import icepop_token_loss
from miles.backends.core_utils.icepop_config import ICEPOP_LOSS, ICEPOP_SCMC_LOSS
from miles.backends.training_utils.loss.hub.losses import get_loss_function


@pytest.mark.parametrize("width", [0, 32, 64, 128])
def test_gradient_matches_reference_ratio_objective_and_mc_correction(width):
    gen = torch.Generator().manual_seed(23)
    logits = torch.randn(4, 7, generator=gen, dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    logq = torch.randn(4, 7, generator=gen, dtype=torch.float64).log_softmax(-1).requires_grad_()
    labels = torch.tensor([0, 1, 2, 3])
    lp = logp[torch.arange(4), labels]
    lq = logq[torch.arange(4), labels]
    advantages = torch.tensor([1.0, -2.0, 0.4, 7.0], requires_grad=True)
    active = torch.tensor([True, True, True, False])
    probes = {}
    if width:
        ids = torch.multinomial(logq.exp(), width, replacement=True, generator=gen)
        probes = dict(
            probe_logp=logp.gather(1, ids),
            probe_logq=logq.gather(1, ids),
            probe_valid=torch.ones_like(ids, dtype=torch.bool),
        )
    actual, _ = icepop_token_loss(lp, lq, advantages, active, **probes)
    expected = []
    for row in range(4):
        ratio = (lp[row] - lq[row].detach()).exp()
        keep = bool(active[row]) and 0.2 <= ratio.detach().item() <= 5.0
        value = -advantages[row].detach() * ratio if keep else 0.0 * lp[row]
        if width and active[row]:
            terms = []
            for token in ids[row].tolist():
                weight = (logp[row, token] - logq[row, token]).detach().exp()
                terms.append(weight * logp[row, token] if 0.2 <= weight <= 5 else 0.0 * logp[row, token])
            value = value + advantages[row].detach() * torch.stack(terms).mean()
        expected.append(value)
    torch.testing.assert_close(actual, torch.stack(expected))
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(torch.stack(expected).sum(), logits)[0],
    )
    assert torch.autograd.grad(actual.sum(), (logq, advantages), allow_unused=True) == (None, None)


def test_inclusive_band_and_extreme_rejected_ratios_have_finite_zero_gradients():
    delta = torch.tensor([0.2, 5.0, 0.19, 5.01], dtype=torch.float64).log()
    lp = torch.cat((delta, torch.tensor([1000.0, -1000.0, float("nan")]))).requires_grad_()
    active = torch.tensor([True] * 6 + [False])
    loss, metrics = icepop_token_loss(lp, torch.zeros_like(lp), torch.ones_like(lp), active)
    loss.sum().backward()
    torch.testing.assert_close(metrics["icepop_keep_fraction"], torch.tensor([1, 1, 0, 0, 0, 0, 0], dtype=lp.dtype))
    torch.testing.assert_close(lp.grad[:2], torch.tensor([-0.2, -5.0], dtype=lp.dtype))
    assert torch.isfinite(loss).all() and torch.isfinite(lp.grad).all()
    assert lp.grad[2:].count_nonzero() == 0


def test_rejected_sample_still_receives_mc_correction_and_padding_is_ignored():
    lp = torch.tensor([-100.0, float("nan")], requires_grad=True)
    probes = torch.full((2, 32), -1.0, requires_grad=True)
    q = torch.full_like(probes, -1.0)
    q[1] = float("nan")
    valid = torch.ones_like(probes, dtype=torch.bool)
    valid[1] = False
    loss, _ = icepop_token_loss(
        lp,
        torch.tensor([-1.0, float("nan")]),
        torch.tensor([2.0, float("nan")]),
        torch.tensor([True, False]),
        probe_logp=probes,
        probe_logq=q,
        probe_valid=valid,
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert lp.grad.count_nonzero() == 0
    torch.testing.assert_close(probes.grad[0], torch.full((32,), 2 / 32))
    assert probes.grad[1].count_nonzero() == 0
    valid[0, 0] = False
    with pytest.raises(ValueError, match="Every active"):
        icepop_token_loss(
            lp,
            lp.detach(),
            torch.ones(2),
            torch.tensor([True, False]),
            probe_logp=probes,
            probe_logq=q,
            probe_valid=valid,
        )


@pytest.mark.usefixtures("single_rank")
@pytest.mark.parametrize("centered", [False, True])
def test_custom_loss_dispatch_scores_response_positions_and_preserves_masks(centered):
    args = _args(
        loss_type="custom_loss",
        custom_loss_function_path=ICEPOP_SCMC_LOSS if centered else ICEPOP_LOSS,
        rollout_score_mode="mc" if centered else "topk",
        rollout_top_logprobs_num=32 if centered else 0,
        rollout_score_transport="array" if centered else "nested",
        use_miles_router=True,
        score_centering_is="mis",
        score_centering_mis_low=0.2,
    )
    logits = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(3), requires_grad=True)
    batch = dict(
        unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
        total_lengths=[4],
        response_lengths=[2],
        loss_masks=[torch.tensor([1.0, 0.0])],
        rollout_log_probs=[torch.tensor([-2.0, float("nan")])],
        advantages=[torch.tensor([1.0, float("nan")])],
    )
    if centered:
        batch.update(
            rollout_topk_token_ids=[np.tile(np.arange(8, dtype=np.int32), (2, 4))],
            rollout_topk_log_probs=[np.full((2, 32), -2.0, dtype=np.float32)],
            rollout_score_mode=["mc"],
        )
    loss, metrics = get_loss_function(args)(args, batch, logits, torch.sum)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, [0, 2, 3]].count_nonzero() == 0
    assert ("sc_correction" in metrics) == centered
    assert "icepop_keep_fraction" in metrics

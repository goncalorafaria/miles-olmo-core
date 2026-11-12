"""IcePop with an optional IID score correction through the Core custom-loss hook.

The sampled-token objective is -A * (p/q) within an inclusive ratio band.
SCMC adds A * mean(stop_gradient(keep * p/q) * log p) over independent q
probes. Rejected sampled actions still receive that expectation correction.
The correction is not value-centered: its gradient matches the reference,
while the reported loss includes its scalar value. No straight-through
sampled-token surrogate or extra CUDA synchronization is needed.
"""

import torch

from miles.backends.core_utils.icepop_config import validate_icepop_args
from miles.backends.training_utils.loss.hub.scmc_loss import _loss


def _accept(log_ratio, low, high):
    return (log_ratio >= log_ratio.new_tensor(low).log()) & (log_ratio <= log_ratio.new_tensor(high).log())


def icepop_token_loss(
    logp,
    logq,
    advantages,
    active,
    *,
    probe_logp=None,
    probe_logq=None,
    probe_valid=None,
    mode="mis",
    tis_clip=2.0,
    mis_low=0.2,
    mis_high=5.0,
):
    """Compute token losses; shared alignment/reduction stays in the MILES adapter."""
    if mode != "mis":
        raise ValueError("IcePop requires masked importance sampling")
    lp = torch.where(active, logp, 0.0)
    lq = torch.where(active, logq.detach(), 0.0)
    advantage = torch.where(active, advantages.detach(), 0.0)
    delta = lp - lq
    keep = active & _accept(delta.detach(), mis_low, mis_high)
    # Mask before exp: rejected extreme ratios must not yield 0 * inf gradients.
    ratio = torch.where(keep, delta, 0.0).exp()
    loss = -advantage * torch.where(keep, ratio, 0.0)
    metrics = {
        "icepop_keep_fraction": keep.to(lp.dtype),
        "icepop_rejected_fraction": (active & ~keep).to(lp.dtype),
        "icepop_importance_weight": torch.where(keep, ratio.detach(), 0.0),
    }
    if probe_logp is not None:
        if probe_logq is None or probe_valid is None or probe_logp.shape != probe_logq.shape:
            raise ValueError("IcePop SCMC requires aligned probe probabilities and validity arrays")
        if probe_logp.shape != (*logp.shape, probe_logp.shape[-1]) or probe_logp.shape[-1] not in (32, 64, 128):
            raise ValueError("IcePop SCMC requires 32, 64 or 128 probes per position")
        if probe_valid.shape != probe_logp.shape or bool((active[..., None] & ~probe_valid).any()):
            raise ValueError("Every active IcePop SCMC position must contain every independent draw")
        plp = torch.where(active[..., None], probe_logp, 0.0)
        plq = torch.where(active[..., None], probe_logq.detach(), 0.0)
        with torch.no_grad():
            probe_delta = plp - plq
            accepted = active[..., None] & _accept(probe_delta, mis_low, mis_high)
            weights = probe_delta.masked_fill(~accepted, -torch.inf).exp()
        correction = (weights * plp).mean(-1)
        loss = loss + advantage * correction
        metrics.update(sc_correction=correction.detach(), sc_probe_weight_mean=weights.mean(-1))
    return loss, metrics


def icepop_loss_function(args, batch, logits, sum_of_sample_mean):
    validate_icepop_args(args)
    return _loss(args, batch, logits, sum_of_sample_mean, centered=False, token_loss_fn=icepop_token_loss)


def icepop_scmc_loss_function(args, batch, logits, sum_of_sample_mean):
    validate_icepop_args(args)
    return _loss(args, batch, logits, sum_of_sample_mean, centered=True, token_loss_fn=icepop_token_loss)

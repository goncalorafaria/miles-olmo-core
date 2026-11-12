"""MILES score-centering loss with IID Monte Carlo probes."""

import torch

from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks
from miles.backends.training_utils.loss.hub.score_centering.monte_carlo import monte_carlo_loss
from miles.backends.training_utils.loss.hub.score_centering_loss import (
    _candidate_log_probs,
    _local_candidates,
    _regularization,
)


def _loss(args, batch, logits, reduce, *, centered, token_loss_fn=monte_carlo_loss):
    if centered:
        modes = batch.get("rollout_score_mode")
        if not modes or any(mode != "mc" for mode in modes):
            raise ValueError("SCMC loss requires explicitly marked Monte Carlo probes on every sample")
        if getattr(args, "use_sampling_support_replay", False):
            raise ValueError("SCMC currently requires unfiltered sampling")
    probabilities = _candidate_log_probs(args, batch, logits, with_candidates=centered)
    selected = torch.cat(probabilities["selected"])
    active = torch.cat(
        get_local_response_loss_masks(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
            args.qkv_format,
            batch.get("max_seq_lens"),
        )
    ).bool()
    rollout = torch.where(active, torch.cat(batch["rollout_log_probs"]).detach(), 0.0)
    advantages = torch.cat(batch["advantages"])
    probes = {}
    if centered:
        ids = _local_candidates(args, batch, "rollout_topk_token_ids", logits.device)
        head = _local_candidates(args, batch, "rollout_topk_log_probs", logits.device)
        probes = dict(probe_logp=selected[:, 1:], probe_logq=head, probe_valid=ids >= 0)
    token_loss, metrics = token_loss_fn(
        selected[:, 0],
        rollout,
        advantages,
        active,
        **probes,
        mode=args.score_centering_is,
        tis_clip=args.score_centering_tis_clip,
        mis_low=args.score_centering_mis_low,
        mis_high=args.score_centering_mis_high,
    )
    pg_loss = reduce(token_loss)
    entropy = torch.cat(probabilities["entropy"]) if "entropy" in probabilities else None
    kl_log_probs = torch.cat(probabilities["kl_log_probs"]) if args.use_kl_loss else None
    loss, log = _regularization(args, batch, entropy, selected[:, 0], rollout, active, reduce, kl_log_probs)
    loss = loss + pg_loss
    log.update({key: reduce(value).detach() for key, value in metrics.items()})
    log.update(loss=loss.detach(), pg_loss=pg_loss.detach())
    log["train_rollout_logprob_abs_diff"] = reduce(
        torch.where(active, (selected[:, 0].detach() - rollout).abs(), 0.0)
    ).detach()
    return loss, log


def scmc_loss_function(args, batch, logits, sum_of_sample_mean):
    return _loss(args, batch, logits, sum_of_sample_mean, centered=True)

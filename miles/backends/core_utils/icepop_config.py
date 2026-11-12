"""Startup contracts for the opt-in Core IcePop custom loss."""

import math
import os

ICEPOP_LOSS = "miles.backends.core_utils.icepop.icepop_loss_function"
ICEPOP_SCMC_LOSS = "miles.backends.core_utils.icepop.icepop_scmc_loss_function"


def validate_icepop_args(args):
    if getattr(args, "loss_type", None) != "custom_loss":
        return
    path = getattr(args, "custom_loss_function_path", None)
    if path not in (ICEPOP_LOSS, ICEPOP_SCMC_LOSS):
        return
    if args.score_centering_is != "mis":
        raise ValueError("IcePop requires score_centering_is=mis; TIS clipping is a different objective")
    low, high = args.score_centering_mis_low, args.score_centering_mis_high
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("IcePop ratio bounds must be finite with 0 < low <= high")
    if not math.isfinite(args.rollout_temperature) or args.rollout_temperature <= 0:
        raise ValueError("IcePop requires a finite positive rollout temperature")
    if args.rollout_top_k != -1 or args.rollout_top_p != 1:
        raise ValueError("Core IcePop currently requires unfiltered sampling (top_k=-1, top_p=1)")
    if args.advantage_estimator != "grpo":
        raise ValueError("Core IcePop currently requires group-centered GRPO advantages")
    for option in (
        "use_tis",
        "custom_tis_function_path",
        "use_opsm",
        "true_on_policy_mode",
        "recompute_logprobs_via_prefill",
        "sglang_speculative_algorithm",
        "custom_pg_loss_reducer_function_path",
        "multi_lora",
        "use_opd",
        "custom_convert_samples_to_train_data_path",
        "use_sampling_support_replay",
    ):
        if getattr(args, option, None):
            raise ValueError(f"Core IcePop does not support --{option.replace('_', '-')}")
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
        raise ValueError("IcePop requires SGLANG_RETURN_ORIGINAL_LOGPROB=0 on rollout servers")
    k = args.rollout_top_logprobs_num
    if path == ICEPOP_SCMC_LOSS:
        if (
            k not in (32, 64, 128)
            or getattr(args, "rollout_score_mode", "topk") != "mc"
            or getattr(args, "rollout_score_transport", "nested") != "array"
            or not getattr(args, "use_miles_router", False)
        ):
            raise ValueError("IcePop SCMC requires MC32/64/128 array probes and the MILES router")
    elif k:
        raise ValueError("Plain IcePop requires rollout_top_logprobs_num=0; select the SCMC loss to use probes")

"""IcePop startup validation must run before any rollout is launched."""

from argparse import Namespace

import pytest

from miles.backends.core_utils.icepop_config import ICEPOP_LOSS, ICEPOP_SCMC_LOSS, validate_icepop_args
from miles.backends.core_utils.integration import configure_rollout


def _args(**overrides):
    return Namespace(
        **(
            dict(
                fully_async=False,
                loss_type="custom_loss",
                custom_loss_function_path=ICEPOP_LOSS,
                score_centering_is="mis",
                score_centering_mis_low=0.2,
                score_centering_mis_high=5.0,
                rollout_temperature=1.0,
                rollout_top_k=-1,
                rollout_top_p=1.0,
                advantage_estimator="grpo",
                rollout_top_logprobs_num=0,
            )
            | overrides
        )
    )


def test_backend_hook_validates_even_synchronous_icepop_and_does_not_mutate_args():
    args = _args()
    before = vars(args).copy()
    configure_rollout(args, None)
    assert vars(args) == before
    with pytest.raises(ValueError, match="score_centering_is=mis"):
        configure_rollout(_args(score_centering_is="tis"), None)


@pytest.mark.parametrize("width", [32, 64, 128])
def test_mc_configuration(width):
    validate_icepop_args(
        _args(
            custom_loss_function_path=ICEPOP_SCMC_LOSS,
            rollout_top_logprobs_num=width,
            rollout_score_mode="mc",
            rollout_score_transport="array",
            use_miles_router=True,
        )
    )


@pytest.mark.parametrize(
    "override",
    [
        dict(score_centering_mis_low=0.0),
        dict(score_centering_mis_high=float("inf")),
        dict(score_centering_mis_low=6.0),
        dict(rollout_temperature=0.0),
        dict(rollout_top_p=0.9),
        dict(use_tis=True),
        dict(use_sampling_support_replay=True),
        dict(advantage_estimator="reinforce"),
        dict(custom_convert_samples_to_train_data_path="custom.convert"),
        dict(rollout_top_logprobs_num=32),
        dict(custom_loss_function_path=ICEPOP_SCMC_LOSS),
    ],
)
def test_incompatible_configuration_fails(override):
    with pytest.raises(ValueError):
        configure_rollout(_args(**override), None)


def test_original_logprob_environment_fails(monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    with pytest.raises(ValueError, match="SGLANG_RETURN_ORIGINAL_LOGPROB=0"):
        validate_icepop_args(_args())


def test_unrelated_losses_are_untouched():
    validate_icepop_args(Namespace(loss_type="score_centering"))
    validate_icepop_args(Namespace(loss_type="custom_loss", custom_loss_function_path="other.loss"))

"""Run the Transformer-backbone ablations A8, A9, and A10.

The selected ablation is defined by the supplied training configuration. This
entrypoint intentionally relies on the few-shot API from alice_jobs_package at
Git commit 0b94c45 or later instead of vendoring that implementation here.
"""

import argparse
from importlib.util import find_spec

from alice_jobs_package.model_runner import AliceModelRunner
from alice_jobs_package.models.base_transformer import BaseTransforemr
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.utils.project_config import ArgsMode


REQUIRED_FEWSHOT_ARGUMENTS = {
    "fewshot_adaptation_enabled",
    "online_inner_lr",
    "online_inner_steps",
    "online_max_support",
    "online_min_support",
    "online_persist_adaptation",
}


def require_fewshot_api() -> None:
    """Fail early when the installed package predates Transformer few-shot evaluation."""
    missing = []
    if find_spec("alice_jobs_package.models.few_shot_wrapper") is None:
        missing.append("alice_jobs_package.models.few_shot_wrapper")

    if not callable(getattr(AliceModelRunner, "evaluate_fewshot_from_dataloader", None)):
        missing.append("AliceModelRunner.evaluate_fewshot_from_dataloader")

    parser_destinations = {
        action.dest for action in TrainingConfig._get_arg_parser()._actions
    }
    missing.extend(sorted(REQUIRED_FEWSHOT_ARGUMENTS - parser_destinations))

    if missing:
        missing_api = ", ".join(missing)
        raise RuntimeError(
            "Transformer ablations A8-A10 require alice_jobs_package from "
            "Git commit 0b94c45 or later. Missing API: " + missing_api
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer ablations A8-A10")
    parser.add_argument(
        "--training_args_mode",
        type=str,
        default="FILE",
        choices=["FILE", "CMD_LINE"],
    )
    parser.add_argument("--training_args_path", type=str, required=True)
    args = parser.parse_args()

    require_fewshot_api()
    training_config = TrainingConfig(
        args_mode=ArgsMode(args.training_args_mode),
        training_args_path=args.training_args_path,
    )
    AliceModelRunner.train_distributed(training_config, BaseTransforemr)


if __name__ == "__main__":
    main()

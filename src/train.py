"""Validate the 2,500-example dataset and run a Cookbook supervised fine-tune."""

import argparse
import asyncio
import json
from collections import Counter
from contextlib import chdir, closing
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Support both package mode (-m src.train_smoke) and direct script execution.
if __package__:
    from .json_utils import iter_jsonl
    from .scorer_config import COMPLEXITY_VALUES, MODEL_NAME
    from .scorer_schema import build_messages, is_valid_complexity
else:
    from json_utils import iter_jsonl
    from scorer_config import COMPLEXITY_VALUES, MODEL_NAME
    from scorer_schema import build_messages, is_valid_complexity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train_2500.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "eval.jsonl"
CONVERSATION_PATH = PROJECT_ROOT / "data" / "train_2500_tinker.jsonl"
LOG_PATH = PROJECT_ROOT / "runs" / "qwen3_8b_train_2500_epoch1"
RENDERER_NAME = "qwen3"
NUM_EPOCHS = 1
BATCH_SIZE = 16
MAX_LENGTH = 4096
LEARNING_RATE = 1e-4
LORA_RANK = 32
EXAMPLES_PER_CLASS = 500
EXPECTED_EXAMPLES = EXAMPLES_PER_CLASS * len(COMPLEXITY_VALUES)


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and require a nonempty prompt on every record."""
    records: list[dict[str, Any]] = []
    # Keep prompt validation here: held-out records do not need score labels.
    with closing(iter_jsonl(path)) as json_records:
        for line_number, record in json_records:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("prompt"), str)
                or not record["prompt"].strip()
            ):
                raise ValueError(f"{path}:{line_number}: prompt must be a nonempty string")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: dataset contains no examples")
    return records


def validate_training_data(
    train_path: Path, eval_path: Path
) -> list[tuple[str, float]]:
    """Validate all training labels and keep the held-out prompts out of training."""
    records = read_records(train_path)
    if len(records) != EXPECTED_EXAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_EXAMPLES} training examples, found {len(records)}"
        )

    # Compare stripped prompts to catch accidental leading/trailing whitespace too.
    eval_prompts = {record["prompt"].strip() for record in read_records(eval_path)}
    seen_prompts: set[str] = set()
    examples: list[tuple[str, float]] = []
    for index, record in enumerate(records, start=1):
        prompt = record["prompt"]
        prompt_key = prompt.strip()
        if prompt_key in seen_prompts:
            raise ValueError(f"Training example {index}: duplicate prompt: {prompt!r}")
        if prompt_key in eval_prompts:
            raise ValueError(
                f"Training example {index}: prompt also occurs in {eval_path}: {prompt!r}"
            )
        seen_prompts.add(prompt_key)

        score = record.get("complexity")
        if not is_valid_complexity(score):
            raise ValueError(
                f"Training example {index}: complexity must be one of {COMPLEXITY_VALUES}"
            )
        if "level" in record:
            level = record["level"]
            if type(level) is not int or level != COMPLEXITY_VALUES.index(score):
                raise ValueError(f"Training example {index}: level does not match complexity")
        examples.append((prompt, float(score)))

    distribution = Counter(score for _, score in examples)
    expected = {score: EXAMPLES_PER_CLASS for score in COMPLEXITY_VALUES}
    if distribution != expected:
        raise ValueError(
            f"Expected {EXAMPLES_PER_CLASS} examples per class; found {dict(distribution)}"
        )
    return examples


def write_conversations(examples: list[tuple[str, float]], path: Path) -> None:
    """Only the prompt and JSON score are used; reasoning never enters training."""
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for prompt, score in examples:
            messages = build_messages(prompt)
            # Training adds the expected answer; evaluation asks the model for it.
            # Double braces escape literal JSON braces; .2f keeps two decimals.
            messages.append(
                {"role": "assistant", "content": f'{{"complexity":{score:.2f}}}'}
            )
            conversation = {"messages": messages}
            output_file.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def print_run_settings(examples: list[tuple[str, float]]) -> None:
    distribution = Counter(score for _, score in examples)
    print(f"Model: {MODEL_NAME}")
    print(f"Training examples: {len(examples)}")
    print("Class distribution: " + ", ".join(
        f"{score:.2f}: {distribution[score]}" for score in COMPLEXITY_VALUES
    ))
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE:g}")
    print(f"LoRA rank: {LORA_RANK}", flush=True)


def run_training() -> None:
    # Lazy imports let --prepare-only work without the Cookbook or any service calls.
    try:
        from tinker_cookbook.supervised import train
        from tinker_cookbook import checkpoint_utils
        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised.data import FromConversationFileBuilder
        from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    except ImportError as error:
        try:
            cookbook_version = version("tinker-cookbook")
        except PackageNotFoundError:
            cookbook_version = "not installed"
        raise ImportError(
            f"Could not load Tinker Cookbook training modules "
            f"(installed Cookbook version: {cookbook_version}). "
            f"Original import error: {error}. "
            "See docs/train_smoke.md for installation and Python compatibility instructions."
        ) from error

    # Cookbook can resume from existing logs; this run must start from base weights.
    if LOG_PATH.exists() and any(LOG_PATH.iterdir()):
        raise ValueError(
            f"Run directory is not empty: {LOG_PATH}. "
            "Move the previous run elsewhere under runs/ before starting a fresh run."
        )

    # blobfile treats Windows drive prefixes as URL schemes. Relative paths work
    # when Cookbook runs from the project directory; chdir restores the caller's cwd.
    with chdir(PROJECT_ROOT):
        dataset_builder = FromConversationFileBuilder(
            file_path=CONVERSATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
            test_size=0,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer=MODEL_NAME,
                renderer_name=RENDERER_NAME,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            ),
        )
        config = train.Config(
            model_name=MODEL_NAME,
            recipe_name="complexity_scorer_2500",
            renderer_name=RENDERER_NAME,
            dataset_builder=dataset_builder,
            learning_rate=LEARNING_RATE,
            lora_rank=LORA_RANK,
            num_epochs=NUM_EPOCHS,
            log_path=LOG_PATH.relative_to(PROJECT_ROOT).as_posix(),
            eval_every=0,
            infrequent_eval_every=0,
            # Disables periodic saves; Cookbook still saves the final state and sampler.
            save_every=0,
        )
        print(f"Tinker run logs: {LOG_PATH}", flush=True)
        # Cookbook owns the forward/backward loop, logging, and final checkpoint save.
        asyncio.run(train.main(config))
        print(f"Training complete. Tinker run logs written to: {LOG_PATH}")

        # Use the public Cookbook helper, not an assumed return value from train.main.
        checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path, required_key="sampler_path")
    if checkpoint is not None:
        print(f"Final sampler checkpoint: {checkpoint.get('sampler_path')}")
    else:
        print(f"Final sampler checkpoint unavailable; inspect the run logs in {LOG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Validate and write data/train_2500_tinker.jsonl without starting training",
    )
    args = parser.parse_args()

    try:
        examples = validate_training_data(TRAIN_PATH, EVAL_PATH)
        write_conversations(examples, CONVERSATION_PATH)
        print_run_settings(examples)
        print(f"Prepared conversations: {CONVERSATION_PATH}", flush=True)
        if args.prepare_only:
            return
        run_training()
    except (OSError, ValueError, ImportError) as error:
        parser.exit(status=1, message=f"Error: {error}\n")


if __name__ == "__main__":
    main()

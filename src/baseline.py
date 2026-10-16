"""Evaluate the base Qwen model or a fine-tuned sampler checkpoint."""

import argparse
import asyncio
import json
from contextlib import closing
from pathlib import Path
from typing import Any

import tinker

# Support both package mode (-m src.baseline) and direct script execution.
if __package__:
    from .json_utils import iter_jsonl, reject_duplicate_keys
    from .scorer_config import COMPLEXITY_VALUES, SYSTEM_PROMPT
    from .scorer_config import MODEL_NAME as BASE_MODEL
    from .scorer_schema import build_messages, is_valid_complexity
else:
    from json_utils import iter_jsonl, reject_duplicate_keys
    from scorer_config import COMPLEXITY_VALUES, SYSTEM_PROMPT
    from scorer_config import MODEL_NAME as BASE_MODEL
    from scorer_schema import build_messages, is_valid_complexity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "results" / "baseline_qwen3_8b.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "results" / "baseline_qwen3_8b_summary.json"
FINETUNED_OUTPUT_PATH = PROJECT_ROOT / "results" / "finetuned_2500_epoch1_qwen3_8b.jsonl"
FINETUNED_SUMMARY_PATH = PROJECT_ROOT / "results" / "finetuned_2500_epoch1_qwen3_8b_summary.json"
THINKING_ENABLED = False
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 50


def parse_prediction(raw_response: str) -> float | None:
    """Return an allowed score, or None for any invalid response."""
    try:
        response = json.loads(raw_response, object_pairs_hook=reject_duplicate_keys)
    except ValueError:
        return None

    if not isinstance(response, dict) or set(response) != {"complexity"}:
        return None

    value = response["complexity"]
    if not is_valid_complexity(value):
        return None
    return float(value)


def load_examples(input_path: Path, limit: int | None) -> list[tuple[str, float]]:
    """Validate selected records before making any paid sampling requests."""
    examples: list[tuple[str, float]] = []
    # closing() closes the reader (and its file) even if --limit causes a break.
    with closing(iter_jsonl(input_path)) as records:
        for line_number, record in records:
            if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
                raise ValueError(f"{input_path}:{line_number}: prompt must be a string")
            expected = record.get("complexity")
            if not is_valid_complexity(expected):
                raise ValueError(
                    f"{input_path}:{line_number}: complexity must be one of "
                    f"{COMPLEXITY_VALUES}"
                )

            # Keep only the two fields used by the baseline.
            examples.append((record["prompt"], float(expected)))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def calculate_metrics(predictions: list[tuple[float, float | None]]) -> dict[str, Any]:
    """Calculate shared console/JSON metrics; MAE and matrix use valid responses."""
    total = len(predictions)
    valid = 0
    correct = 0
    absolute_error = 0.0
    matrix = [[0 for _ in COMPLEXITY_VALUES] for _ in COMPLEXITY_VALUES]

    for expected, predicted in predictions:
        if predicted is None:
            continue
        valid += 1
        correct += int(predicted == expected)
        absolute_error += abs(predicted - expected)
        row = COMPLEXITY_VALUES.index(expected)
        column = COMPLEXITY_VALUES.index(predicted)
        matrix[row][column] += 1

    # None becomes JSON null when a metric has no applicable examples.
    return {
        "total_examples": total,
        "valid_response_format": valid,
        "invalid_response_format": total - valid,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "exact_accuracy": correct / total if total else None,
        "mean_absolute_error": absolute_error / valid if valid else None,
        "confusion_matrix": {
            "labels": list(COMPLEXITY_VALUES),
            "rows": "expected",
            "columns": "predicted",
            "matrix": matrix,
        },
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print the same metrics saved in the summary file."""
    total = metrics["total_examples"]
    valid = metrics["valid_response_format"]
    print(f"\nTotal examples: {total}")
    print(f"Valid response format: {valid}")
    print(f"Invalid response format: {metrics['invalid_response_format']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Incorrect predictions: {metrics['incorrect_predictions']}")
    if total:
        print(f"Exact accuracy: {metrics['exact_accuracy']:.2%}")
    else:
        print("Exact accuracy: N/A (no examples)")
    if valid:
        print(f"Mean absolute error: {metrics['mean_absolute_error']:.4f}")
    else:
        print("Mean absolute error: N/A (no valid predictions)")

    print("Confusion matrix (rows = expected, columns = predicted; valid only):")
    print("       " + " ".join(f"{score:>5.2f}" for score in COMPLEXITY_VALUES))
    for score, counts in zip(COMPLEXITY_VALUES, metrics["confusion_matrix"]["matrix"]):
        print(f"{score:>5.2f}  " + " ".join(f"{count:>5}" for count in counts))


def write_summary(
    metrics: dict[str, Any], input_path: Path, limit: int | None,
    checkpoint: str | None = None, summary_path: Path = SUMMARY_PATH,
) -> None:
    """Save metrics and the configuration used for this completed run."""
    dataset_path = input_path.resolve()
    try:
        dataset_path = dataset_path.relative_to(PROJECT_ROOT)
    except ValueError:
        pass  # Datasets outside the project keep their absolute path.

    summary = {
        "model": BASE_MODEL,
        "dataset": dataset_path.as_posix(),
        "thinking_enabled": THINKING_ENABLED,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        **metrics,
    }
    if limit is not None:
        summary["limit"] = limit
    if checkpoint is not None:
        summary["checkpoint"] = checkpoint

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)
        summary_file.write("\n")


async def evaluate(
    examples: list[tuple[str, float]], input_path: Path, limit: int | None,
    checkpoint: str | None = None,
) -> None:
    """Reuse one Tinker sampling client and save each response immediately."""
    predictions: list[tuple[float, float | None]] = []
    output_path = FINETUNED_OUTPUT_PATH if checkpoint is not None else OUTPUT_PATH
    summary_path = FINETUNED_SUMMARY_PATH if checkpoint is not None else SUMMARY_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse one service and sampling client for every example in this run.
    service = tinker.ServiceClient()
    if checkpoint is not None:
        # model_path loads the trained adapter; base_model alone loads untouched weights.
        print(f"Evaluating sampler checkpoint: {checkpoint}", flush=True)
        client = await service.create_sampling_client_async(model_path=checkpoint)
    else:
        client = await service.create_sampling_client_async(base_model=BASE_MODEL)
    tokenizer = client.get_tokenizer()
    sampling_params = tinker.types.SamplingParams(
        max_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE
    )

    # Each run replaces the output file; flush every row to preserve partial runs.
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, (user_prompt, expected) in enumerate(examples, start=1):
            tokens = tokenizer.apply_chat_template(
                build_messages(user_prompt),
                add_generation_prompt=True,
                enable_thinking=THINKING_ENABLED,
                return_dict=False,
            )
            result = await client.sample_async(
                prompt=tinker.types.ModelInput.from_ints(tokens),
                num_samples=1,
                sampling_params=sampling_params,
            )
            raw_response = tokenizer.decode(
                result.sequences[0].tokens,
                skip_special_tokens=True,
            )
            predicted = parse_prediction(raw_response)
            record = {
                "prompt": user_prompt,
                "expected": expected,
                "predicted": predicted,
                "valid": predicted is not None,
                "raw_response": raw_response,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            predictions.append((expected, predicted))
            label = "invalid" if predicted is None else f"{predicted:.2f}"
            print(f"[{index}/{len(examples)}] expected={expected:.2f}, predicted={label}")

    metrics = calculate_metrics(predictions)
    write_summary(metrics, input_path, limit, checkpoint=checkpoint, summary_path=summary_path)
    print_metrics(metrics)
    print(f"Results: {output_path}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "eval.jsonl",
        help="Input JSONL dataset (default: data/eval.jsonl)",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples")
    parser.add_argument(
        "--checkpoint",
        help="Tinker sampler checkpoint path printed by training (default: untouched base model)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.checkpoint is not None and (
        not args.checkpoint.startswith("tinker://")
        or "/sampler_weights/" not in args.checkpoint
        or not args.checkpoint.rsplit("/", 1)[-1]
    ):
        parser.error("--checkpoint must be a tinker://.../sampler_weights/... path")

    try:
        examples = load_examples(args.input, args.limit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not examples:
        parser.error("The input dataset contains no examples")

    # asyncio.run starts the event loop used by Tinker's async APIs.
    asyncio.run(evaluate(examples, args.input, args.limit, checkpoint=args.checkpoint))


if __name__ == "__main__":
    main()

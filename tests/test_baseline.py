"""Verify checkpoint selection and result separation without paid sampling calls."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from src import baseline


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def check_evaluation(self, checkpoint: str | None) -> None:
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [1, 2]
        tokenizer.decode.return_value = '{"complexity":0.25}'
        client = Mock()
        client.get_tokenizer.return_value = tokenizer
        client.sample_async = AsyncMock(
            return_value=SimpleNamespace(sequences=[SimpleNamespace(tokens=[3])])
        )
        service = Mock()
        service.create_sampling_client_async = AsyncMock(return_value=client)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "baseline.jsonl"
            summary_path = root / "baseline_summary.json"
            finetuned_output_path = root / "finetuned.jsonl"
            finetuned_summary_path = root / "finetuned_summary.json"
            # An existing baseline must survive evaluating trained weights.
            output_path.write_text("original baseline", encoding="utf-8")
            summary_path.write_text("original summary", encoding="utf-8")
            with (
                patch.object(baseline.tinker, "ServiceClient", return_value=service),
                patch.object(baseline, "OUTPUT_PATH", output_path),
                patch.object(baseline, "SUMMARY_PATH", summary_path),
                patch.object(baseline, "FINETUNED_OUTPUT_PATH", finetuned_output_path),
                patch.object(baseline, "FINETUNED_SUMMARY_PATH", finetuned_summary_path),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                await baseline.evaluate(
                    [("Example prompt", 0.25)], root / "eval.jsonl", None, checkpoint
                )
            if checkpoint is None:
                service.create_sampling_client_async.assert_awaited_once_with(
                    base_model=baseline.BASE_MODEL
                )
                self.assertFalse(finetuned_output_path.exists())
                self.assertFalse(finetuned_summary_path.exists())
            else:
                service.create_sampling_client_async.assert_awaited_once_with(
                    model_path=checkpoint
                )
                self.assertEqual(output_path.read_text(encoding="utf-8"), "original baseline")
                self.assertEqual(summary_path.read_text(encoding="utf-8"), "original summary")
                output_path = finetuned_output_path
                summary_path = finetuned_summary_path
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["predicted"], 0.25)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["exact_accuracy"], 1.0)
            if checkpoint is None:
                self.assertNotIn("checkpoint", summary)
            else:
                self.assertEqual(summary["checkpoint"], checkpoint)

        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": baseline.SYSTEM_PROMPT},
                {"role": "user", "content": "Example prompt"},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )

    async def test_default_uses_untouched_model_and_original_output_files(self) -> None:
        await self.check_evaluation(None)

    async def test_checkpoint_loads_trained_weights_and_preserves_baseline(self) -> None:
        await self.check_evaluation("tinker://example-run/sampler_weights/final")

    def test_invalid_checkpoint_is_rejected_before_service_calls(self) -> None:
        for checkpoint in ("", "Qwen/Qwen3-8B", "tinker://run/weights/final"):
            with (
                self.subTest(checkpoint=checkpoint),
                patch("sys.argv", ["baseline.py", "--checkpoint", checkpoint]),
                patch.object(baseline.tinker, "ServiceClient") as service,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    baseline.main()
                self.assertEqual(error.exception.code, 2)
                service.assert_not_called()


if __name__ == "__main__":
    unittest.main()

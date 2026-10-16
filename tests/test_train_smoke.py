"""Offline checks for label validation, held-out separation, and target format."""

import contextlib
import builtins
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import train_smoke
from src.scorer_config import COMPLEXITY_VALUES, SYSTEM_PROMPT


class TrainingDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        directory = Path(self.temp_directory.name)
        self.train_path = directory / "train.jsonl"
        self.eval_path = directory / "eval.jsonl"
        self.output_path = directory / "train_tinker.jsonl"
        self.records = [
            {
                "prompt": f"Training prompt {level}-{index}: Czech č, quote \" and newline\n",
                "complexity": score,
                "level": level,
                "reasoning": "PRIVATE_REASONING_MUST_NOT_APPEAR",
            }
            for level, score in enumerate(COMPLEXITY_VALUES)
            for index in range(500)
        ]
        self.save_records(self.train_path, self.records)
        self.save_records(self.eval_path, [{"prompt": "Held-out prompt"}])

    @staticmethod
    def save_records(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def validate(self) -> list[tuple[str, float]]:
        return train_smoke.validate_training_data(self.train_path, self.eval_path)

    def test_balanced_data_produces_only_json_assistant_targets(self) -> None:
        examples = self.validate()
        train_smoke.write_conversations(examples, self.output_path)
        output = self.output_path.read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_REASONING_MUST_NOT_APPEAR", output)
        conversations = [json.loads(line) for line in output.splitlines()]
        self.assertEqual(len(conversations), 2500)
        for record, conversation in zip(self.records, conversations):
            self.assertEqual(set(conversation), {"messages"})
            system, user, assistant = conversation["messages"]
            self.assertEqual(system, {"role": "system", "content": SYSTEM_PROMPT})
            self.assertEqual(user, {"role": "user", "content": record["prompt"]})
            self.assertEqual(assistant["role"], "assistant")
            self.assertEqual(json.loads(assistant["content"]), {"complexity": record["complexity"]})
        self.assertEqual(
            {conversation["messages"][2]["content"] for conversation in conversations},
            {'{"complexity":0.00}', '{"complexity":0.25}', '{"complexity":0.50}',
             '{"complexity":0.75}', '{"complexity":1.00}'},
        )

    def test_wrong_example_count_is_rejected(self) -> None:
        self.save_records(self.train_path, self.records[:-1])
        with self.assertRaisesRegex(ValueError, "Expected 2500"):
            self.validate()

    def test_unbalanced_classes_are_rejected(self) -> None:
        self.records[0].update(complexity=0.25, level=1)
        self.save_records(self.train_path, self.records)
        with self.assertRaisesRegex(ValueError, "500 examples per class"):
            self.validate()

    def test_invalid_scores_are_rejected(self) -> None:
        for score in (True, False, "0.25", None, 0.1, float("nan"), float("inf")):
            with self.subTest(score=score):
                self.records[0]["complexity"] = score
                self.save_records(self.train_path, self.records)
                with self.assertRaisesRegex(ValueError, "complexity must be one of"):
                    self.validate()

    def test_inconsistent_level_is_rejected(self) -> None:
        for level in (True, 0.0, 1):
            with self.subTest(level=level):
                self.records[0]["level"] = level
                self.save_records(self.train_path, self.records)
                with self.assertRaisesRegex(ValueError, "level does not match"):
                    self.validate()

    def test_duplicate_prompts_are_rejected(self) -> None:
        self.records[1]["prompt"] = "  " + self.records[0]["prompt"] + "  "
        self.save_records(self.train_path, self.records)
        with self.assertRaisesRegex(ValueError, "duplicate prompt"):
            self.validate()

    def test_held_out_overlap_is_rejected(self) -> None:
        self.save_records(self.eval_path, [{"prompt": self.records[0]["prompt"].strip()}])
        with self.assertRaisesRegex(ValueError, "also occurs"):
            self.validate()

    def test_malformed_json_and_duplicate_keys_are_rejected(self) -> None:
        for line in ('{"prompt":', '{"prompt":"a","prompt":"b"}'):
            with self.subTest(line=line):
                self.train_path.write_text(line, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid JSON"):
                    self.validate()

    def test_invalid_prompts_are_rejected(self) -> None:
        for prompt in (None, 123, "", " \n "):
            with self.subTest(prompt=prompt):
                self.records[0]["prompt"] = prompt
                self.save_records(self.train_path, self.records)
                with self.assertRaisesRegex(ValueError, "nonempty string"):
                    self.validate()

    def test_missing_or_empty_benchmark_is_rejected(self) -> None:
        self.eval_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()
        self.eval_path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no examples"):
            self.validate()

    def test_prepare_only_never_starts_training(self) -> None:
        with (
            patch.object(train_smoke, "TRAIN_PATH", self.train_path),
            patch.object(train_smoke, "EVAL_PATH", self.eval_path),
            patch.object(train_smoke, "CONVERSATION_PATH", self.output_path),
            patch.object(train_smoke, "run_training") as run_training,
            patch("sys.argv", ["train_smoke.py", "--prepare-only"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train_smoke.main()
            run_training.assert_not_called()
        self.assertTrue(self.output_path.exists())

    def test_invalid_data_does_not_replace_existing_output(self) -> None:
        self.output_path.write_text("previous output", encoding="utf-8")
        self.save_records(self.train_path, self.records[:-1])
        with (
            patch.object(train_smoke, "TRAIN_PATH", self.train_path),
            patch.object(train_smoke, "EVAL_PATH", self.eval_path),
            patch.object(train_smoke, "CONVERSATION_PATH", self.output_path),
            patch.object(train_smoke, "run_training") as run_training,
            patch("sys.argv", ["train_smoke.py"]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                train_smoke.main()
            self.assertEqual(error.exception.code, 1)
            run_training.assert_not_called()
        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "previous output")


class CookbookImportTests(unittest.TestCase):
    def test_error_preserves_missing_module_and_actual_package_version(self) -> None:
        original_import = builtins.__import__

        def fail_cookbook_import(name, *args, **kwargs):
            if name == "tinker_cookbook.supervised":
                raise ModuleNotFoundError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fail_cookbook_import),
            patch.object(train_smoke, "version", return_value="0.5.7"),
        ):
            with self.assertRaises(ImportError) as error:
                train_smoke.run_training()
        self.assertIn("installed Cookbook version: 0.5.7", str(error.exception))
        self.assertIn("No module named 'torch'", str(error.exception))
        self.assertNotIn("empty tinker-cookbook 0.0.0", str(error.exception))


if __name__ == "__main__":
    unittest.main()

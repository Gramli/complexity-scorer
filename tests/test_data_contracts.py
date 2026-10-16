"""Protect the existing reader and prediction rules during shared-code changes."""

import json
import tempfile
import unittest
from pathlib import Path

from src import baseline, train_smoke
from src.scorer_config import COMPLEXITY_VALUES


class PredictionParsingTests(unittest.TestCase):
    def test_allowed_scores_accept_numbers_and_return_floats(self) -> None:
        for score in (*COMPLEXITY_VALUES, 0, 1):
            with self.subTest(score=score):
                result = baseline.parse_prediction(json.dumps({"complexity": score}))
                self.assertEqual(result, score)
                self.assertIs(type(result), float)

    def test_invalid_scores_return_none(self) -> None:
        invalid_scores = (
            True, False, "0.25", None, 0.1,
            float("nan"), float("inf"), float("-inf"), [], {},
        )
        for score in invalid_scores:
            with self.subTest(score=score):
                self.assertIsNone(
                    baseline.parse_prediction(json.dumps({"complexity": score}))
                )

    def test_invalid_json_shapes_and_duplicate_properties_return_none(self) -> None:
        responses = (
            "", "not JSON", "null", "[]", "0.5", "{}",
            '{"complexity":0.25,"reasoning":"extra property"}',
            '{"complexity":0.25,"complexity":0.50}',
            '{"complexity":{"score":0.25,"score":0.50}}',
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIsNone(baseline.parse_prediction(response))


class DatasetReadingTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "examples.jsonl"

    def test_readers_preserve_prompt_text_and_keep_their_different_outputs(self) -> None:
        record = {
            "prompt": '  Czech č, quote " and newline\n',
            "complexity": 1,
            "level": 4,
            "reasoning": "metadata",
        }
        self.path.write_text("\n \n" + json.dumps(record) + "\n", encoding="utf-8")

        self.assertEqual(baseline.load_examples(self.path, None), [(record["prompt"], 1.0)])
        self.assertEqual(train_smoke.read_records(self.path), [record])

    def test_evaluation_limit_stops_before_invalid_unselected_record(self) -> None:
        self.path.write_text(
            '\n{"prompt":"Selected","complexity":0.25}\n\nnot JSON\n',
            encoding="utf-8",
        )
        self.assertEqual(baseline.load_examples(self.path, 1), [("Selected", 0.25)])
        with self.assertRaises(ValueError) as error:
            baseline.load_examples(self.path, None)
        self.assertTrue(str(error.exception).startswith(f"{self.path}:4: invalid JSON:"))

    def test_empty_files_keep_their_different_behaviors(self) -> None:
        for content in ("", "\n \n"):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                self.assertEqual(baseline.load_examples(self.path, None), [])
                with self.assertRaisesRegex(ValueError, "dataset contains no examples"):
                    train_smoke.read_records(self.path)

    def test_evaluation_allows_empty_prompts_but_training_rejects_them(self) -> None:
        for prompt in ("", " \n "):
            with self.subTest(prompt=prompt):
                self.path.write_text(
                    json.dumps({"prompt": prompt, "complexity": 0}), encoding="utf-8"
                )
                self.assertEqual(baseline.load_examples(self.path, None), [(prompt, 0.0)])
                with self.assertRaisesRegex(ValueError, "nonempty string"):
                    train_smoke.read_records(self.path)

    def test_held_out_records_need_only_a_prompt(self) -> None:
        self.path.write_text('{"prompt":"Held out"}\n', encoding="utf-8")
        self.assertEqual(train_smoke.read_records(self.path), [{"prompt": "Held out"}])
        with self.assertRaisesRegex(ValueError, "complexity must be one of"):
            baseline.load_examples(self.path, None)

    def test_evaluation_rejects_invalid_labels(self) -> None:
        for score in (True, False, "0.25", None, 0.1, float("nan"), float("inf")):
            with self.subTest(score=score):
                self.path.write_text(
                    json.dumps({"prompt": "Example", "complexity": score}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "complexity must be one of"):
                    baseline.load_examples(self.path, None)

    def test_readers_report_physical_line_numbers_for_invalid_records(self) -> None:
        invalid_lines = (
            '{"prompt":',
            '{"prompt":"First","prompt":"Second","complexity":0.25}',
            '{"prompt":"Example","complexity":0.25,"metadata":{"id":1,"id":2}}',
        )
        for line in invalid_lines:
            with self.subTest(line=line):
                self.path.write_text("\n \n" + line, encoding="utf-8")
                with self.assertRaises(ValueError) as evaluation_error:
                    baseline.load_examples(self.path, None)
                with self.assertRaises(ValueError) as training_error:
                    train_smoke.read_records(self.path)

                expected_prefix = f"{self.path}:3: invalid JSON:"
                self.assertTrue(str(evaluation_error.exception).startswith(expected_prefix))
                self.assertTrue(str(training_error.exception).startswith(expected_prefix))

    def test_readers_reject_non_objects_and_non_string_prompts(self) -> None:
        for record in ([], None, 123, {}, {"prompt": None}, {"prompt": 123}):
            with self.subTest(record=record):
                self.path.write_text("\n" + json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError) as evaluation_error:
                    baseline.load_examples(self.path, None)
                self.assertEqual(
                    str(evaluation_error.exception), f"{self.path}:2: prompt must be a string"
                )
                with self.assertRaises(ValueError) as training_error:
                    train_smoke.read_records(self.path)
                self.assertEqual(
                    str(training_error.exception),
                    f"{self.path}:2: prompt must be a nonempty string",
                )


if __name__ == "__main__":
    unittest.main()

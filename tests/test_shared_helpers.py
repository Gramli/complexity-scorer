"""Check file cleanup and message isolation at the shared-code boundaries."""

import io
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from src import baseline, train_smoke
from src.json_utils import iter_jsonl
from src.scorer_config import SYSTEM_PROMPT
from src.scorer_schema import build_messages


class SharedHelperTests(unittest.TestCase):
    def test_jsonl_reader_leaves_record_shape_validation_to_callers(self) -> None:
        input_file = io.StringIO('\n[1, 2]\nnull\n{"prompt":"Example"}\n')
        with patch.object(Path, "open", return_value=input_file):
            with closing(iter_jsonl(Path("unused.jsonl"))) as records:
                self.assertEqual(
                    list(records), [(2, [1, 2]), (3, None), (4, {"prompt": "Example"})]
                )
        self.assertTrue(input_file.closed)

    def test_evaluation_closes_file_when_limit_stops_reading(self) -> None:
        input_file = io.StringIO('{"prompt":"Example","complexity":0.25}\ninvalid JSON')
        with patch.object(Path, "open", return_value=input_file):
            examples = baseline.load_examples(Path("unused.jsonl"), 1)
        self.assertEqual(examples, [("Example", 0.25)])
        self.assertTrue(input_file.closed)

    def test_both_loaders_close_file_when_prompt_validation_fails(self) -> None:
        evaluation_file = io.StringIO('{"prompt":123,"complexity":0.25}\n')
        with patch.object(Path, "open", return_value=evaluation_file):
            with self.assertRaises(ValueError):
                baseline.load_examples(Path("unused.jsonl"), None)
        self.assertTrue(evaluation_file.closed)

        training_file = io.StringIO('{"prompt":123}\n')
        with patch.object(Path, "open", return_value=training_file):
            with self.assertRaises(ValueError):
                train_smoke.read_records(Path("unused.jsonl"))
        self.assertTrue(training_file.closed)

    def test_changing_one_conversation_does_not_affect_another(self) -> None:
        training_messages = build_messages("Training prompt")
        training_messages[0]["content"] = "Changed system prompt"
        training_messages.append({"role": "assistant", "content": '{"complexity":0.25}'})

        self.assertEqual(
            build_messages("  Evaluation prompt\n"),
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "  Evaluation prompt\n"},
            ],
        )


if __name__ == "__main__":
    unittest.main()

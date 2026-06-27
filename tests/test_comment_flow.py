import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import comment


class CommentFlowTest(unittest.TestCase):
    def test_ask_memory_dispatch_uses_fake_sub_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "2026-06-23-20-18-02"
            target.mkdir()
            (target / "diary.txt").write_text("今天见到了 Sarah。", encoding="utf-8")

            tool_call = {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "ask_memory",
                    "arguments": '{"question": "Sarah 之前发生过什么？"}',
                },
            }
            fake_result = {
                "question": "Sarah 之前发生过什么？",
                "answer": 'Answer: 有旧背景。\nEvidence: “见到了 Sarah” [2026-06-23-20-18-02 diary line:1-1]',
                "tool_calls": [],
                "tool_call_count": 0,
            }

            memory_questions = []
            with patch.object(comment, "run_memory_agent", return_value=fake_result):
                result = comment.run_main_tool_call(tool_call, target, root, memory_questions)

            self.assertEqual(result["question"], "Sarah 之前发生过什么？")
            self.assertIn("旧背景", result["answer"])
            self.assertEqual(len(memory_questions), 1)

    def test_read_entry_lines_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "2026-06-23-20-18-02"
            target.mkdir()
            (target / "diary.txt").write_text("第一行\n第二行", encoding="utf-8")

            tool_call = {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "read_entry_lines",
                    "arguments": '{"diary_id": "2026-06-23-20-18-02", "start_line": 2, "end_line": 2}',
                },
            }
            result = comment.run_main_tool_call(tool_call, target, root, [])
            self.assertEqual(result["lines"][0]["text"], "第二行")


if __name__ == "__main__":
    unittest.main()

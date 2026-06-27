import tempfile
import unittest
from pathlib import Path

from agent_tools import (
    list_diaries,
    read_entry_lines,
    search_keyword,
    search_regex,
    search_title,
)


def make_entry(root: Path, diary_id: str, diary: str, title: str = "", comment: str = "") -> None:
    entry = root / diary_id
    entry.mkdir()
    (entry / "diary.txt").write_text(diary, encoding="utf-8")
    if title:
        (entry / "title.txt").write_text(title, encoding="utf-8")
    if comment:
        (entry / "comment.txt").write_text(comment, encoding="utf-8")


class AgentToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_entry(
            self.root,
            "2026-06-23-20-18-02",
            "第一行 Sarah 来了\n\n第三行 需要工作\n第四行 Cadence said see you tomorrow",
            "Sarah 和 lab 的一天",
            "旧评论第一行\n第二行 comment 太像说教",
        )
        make_entry(
            self.root,
            "2026-06-24-10-00-00",
            "今天没有关键词\nformal verification 有点难",
            "FV 工作",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_keyword_search_is_case_insensitive_and_line_based(self) -> None:
        result = search_keyword(self.root, "diary.txt", "comment.txt", "sarah")
        self.assertEqual(len(result["results"]), 1)
        hit = result["results"][0]
        self.assertEqual(hit["diary_id"], "2026-06-23-20-18-02")
        self.assertEqual(hit["line_start"], 1)
        self.assertEqual(hit["ref"], "2026-06-23-20-18-02 diary line:1-1")
        self.assertIn("Sarah", hit["quote"])

    def test_regex_can_search_comments(self) -> None:
        result = search_regex(self.root, "diary.txt", "comment.txt", "comment.*说教", "comment")
        self.assertEqual(len(result["results"]), 1)
        hit = result["results"][0]
        self.assertEqual(hit["diary_or_comment"], "comment")
        self.assertEqual(hit["line_start"], 2)

    def test_read_entry_lines_uses_real_file_lines(self) -> None:
        result = read_entry_lines(self.root, "diary.txt", "comment.txt", "2026-06-23-20-18-02", 1, 3)
        self.assertEqual(result["line_start"], 1)
        self.assertEqual(result["line_end"], 3)
        self.assertEqual(result["lines"][1]["line"], 2)
        self.assertEqual(result["lines"][1]["text"], "")
        self.assertEqual(result["lines"][2]["text"], "第三行 需要工作")

    def test_title_search_and_list_diaries(self) -> None:
        title_result = search_title(self.root, "title.txt", "lab")
        self.assertEqual(title_result["results"][0]["title"], "Sarah 和 lab 的一天")

        listed = list_diaries(self.root, "diary.txt", "comment.txt", "title.txt", order="asc")
        self.assertEqual([item["diary_id"] for item in listed["results"]], [
            "2026-06-23-20-18-02",
            "2026-06-24-10-00-00",
        ])
        self.assertTrue(listed["results"][0]["has_comment"])
        self.assertFalse(listed["results"][1]["has_comment"])


if __name__ == "__main__":
    unittest.main()

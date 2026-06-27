import tempfile
import unittest
from pathlib import Path

from agent_tools import (
    list_diaries,
    read_entry_lines,
    search_text,
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
            "2026-06-22-20-00-00",
            "first Sarah line\nsecond Sarah lab line\nthird ordinary line",
            "Sarah and lab",
            "old comment line\nsecond comment sounds preachy",
        )
        make_entry(
            self.root,
            "2026-06-23-20-18-02",
            "target Sarah line\nformal verification is hard",
            "target day",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_search_text_literal_is_case_insensitive_and_line_based(self) -> None:
        result = search_text(self.root, "diary.txt", "comment.txt", "sarah", "literal")
        self.assertEqual(result["total_matches"], 3)
        self.assertEqual(result["returned_count"], 3)
        hit = result["results"][0]
        self.assertEqual(hit["diary_id"], "2026-06-22-20-00-00")
        self.assertEqual(hit["line_start"], 1)
        self.assertEqual(hit["ref"], "2026-06-22-20-00-00 diary line:1-1")
        self.assertIn("Sarah", hit["quote"])

    def test_search_text_regex_can_search_comments(self) -> None:
        result = search_text(self.root, "diary.txt", "comment.txt", "comment.*preachy", "regex", "comment")
        self.assertEqual(result["total_matches"], 1)
        hit = result["results"][0]
        self.assertEqual(hit["diary_or_comment"], "comment")
        self.assertEqual(hit["line_start"], 2)

    def test_search_text_reports_truncation_and_total_matches(self) -> None:
        result = search_text(self.root, "diary.txt", "comment.txt", "Sarah", "literal", limit=1)
        self.assertEqual(result["total_matches"], 3)
        self.assertEqual(result["returned_count"], 1)
        self.assertTrue(result["truncated"])
        self.assertIn("Narrow", result["suggestion"])

    def test_search_text_excludes_target_and_future_when_target_id_is_given(self) -> None:
        result = search_text(
            self.root,
            "diary.txt",
            "comment.txt",
            "Sarah",
            "literal",
            target_id="2026-06-23-20-18-02",
        )
        self.assertEqual(result["total_matches"], 2)
        self.assertTrue(all(item["diary_id"] < "2026-06-23-20-18-02" for item in result["results"]))

    def test_read_entry_lines_uses_real_file_lines(self) -> None:
        result = read_entry_lines(self.root, "diary.txt", "comment.txt", "2026-06-22-20-00-00", 1, 3)
        self.assertEqual(result["line_start"], 1)
        self.assertEqual(result["line_end"], 3)
        self.assertEqual(result["lines"][1]["line"], 2)
        self.assertEqual(result["lines"][1]["text"], "second Sarah lab line")

    def test_title_search_and_list_diaries_respect_target_scope(self) -> None:
        title_result = search_title(self.root, "title.txt", "target", target_id="2026-06-23-20-18-02")
        self.assertEqual(title_result["total_matches"], 0)

        listed = list_diaries(self.root, "diary.txt", "comment.txt", "title.txt", order="asc", target_id="2026-06-23-20-18-02")
        self.assertEqual([item["diary_id"] for item in listed["results"]], ["2026-06-22-20-00-00"])
        self.assertTrue(listed["results"][0]["has_comment"])


if __name__ == "__main__":
    unittest.main()

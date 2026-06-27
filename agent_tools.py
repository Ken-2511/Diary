import json
import re
from pathlib import Path


DIARY_ID_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")
VALID_SOURCES = {"diary", "comment"}
PUNCTUATION = "。！？!?；;，,"

MAIN_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "ask_memory",
            "description": "Ask a memory research sub-agent a natural-language question about past diaries. The sub-agent searches diary/comment lines and returns a concise answer with short evidence quotes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_entry_lines",
            "description": "Read exact lines from a diary or comment by diary id. Line numbers are real file lines, 1-based and inclusive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diary_id": {"type": "string"},
                    "diary_or_comment": {"type": "string", "enum": ["diary", "comment"], "default": "diary"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["diary_id", "start_line", "end_line"],
            },
        },
    },
]

RESEARCH_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_diaries",
            "description": "List diary folders with metadata. Use date filters to inspect recent or historical periods before searching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": ["string", "null"], "default": None},
                    "end_date": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": 30},
                    "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_title",
            "description": "Search diary titles by case-insensitive literal text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "start_date": {"type": ["string", "null"], "default": None},
                    "end_date": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "Search diary or comment lines by case-insensitive literal text. Returns line-level results with a short quote and full line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "diary_or_comment": {"type": "string", "enum": ["diary", "comment"], "default": "diary"},
                    "start_date": {"type": ["string", "null"], "default": None},
                    "end_date": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_regex",
            "description": "Search diary or comment lines by Python regex with case-insensitive matching. Returns line-level results with a short quote and full line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "diary_or_comment": {"type": "string", "enum": ["diary", "comment"], "default": "diary"},
                    "start_date": {"type": ["string", "null"], "default": None},
                    "end_date": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["pattern"],
            },
        },
    },
    MAIN_SCHEMAS[1],
]


def diary_dirs(diary_root: Path) -> list[Path]:
    if not diary_root.exists():
        return []
    return sorted(p for p in diary_root.iterdir() if p.is_dir() and DIARY_ID_RE.fullmatch(p.name))


def entry_path(diary_root: Path, diary_id: str, diary_name: str, comment_name: str, diary_or_comment: str) -> Path:
    if not DIARY_ID_RE.fullmatch(diary_id):
        raise ValueError(f"Invalid diary id: {diary_id}")
    if diary_or_comment not in VALID_SOURCES:
        raise ValueError(f"Invalid source: {diary_or_comment}")
    filename = diary_name if diary_or_comment == "diary" else comment_name
    return diary_root / diary_id / filename


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return len(read_lines(path))


def title_for(diary_dir: Path, title_name: str) -> str | None:
    path = diary_dir / title_name
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def in_date_range(diary_id: str, start_date: str | None, end_date: str | None) -> bool:
    date = diary_id[:10]
    return (start_date is None or date >= start_date) and (end_date is None or date <= end_date)


def make_ref(diary_id: str, diary_or_comment: str, start_line: int, end_line: int) -> str:
    return f"{diary_id} {diary_or_comment} line:{start_line}-{end_line}"


def extract_quote(line: str, start: int, end: int, max_chars: int = 80) -> str:
    if not line:
        return ""
    left_candidates = [line.rfind(mark, 0, start) for mark in PUNCTUATION]
    right_candidates = [line.find(mark, end) for mark in PUNCTUATION]
    left = max(left_candidates)
    right_values = [i for i in right_candidates if i != -1]
    quote_start = left + 1 if left != -1 else max(0, start - max_chars // 2)
    quote_end = (min(right_values) + 1) if right_values else min(len(line), end + max_chars // 2)
    if quote_end - quote_start > max_chars:
        extra = max_chars - (end - start)
        quote_start = max(0, start - extra // 2)
        quote_end = min(len(line), end + extra // 2)
    return line[quote_start:quote_end].strip()


def format_line_result(diary_id: str, diary_or_comment: str, line_number: int, line: str, start: int, end: int) -> dict:
    return {
        "diary_id": diary_id,
        "date": diary_id[:10],
        "diary_or_comment": diary_or_comment,
        "line_start": line_number,
        "line_end": line_number,
        "matched_text": line[start:end],
        "quote": extract_quote(line, start, end),
        "line_text": line,
        "ref": make_ref(diary_id, diary_or_comment, line_number, line_number),
    }


def list_diaries(
    diary_root: Path,
    diary_name: str,
    comment_name: str,
    title_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 30,
    order: str = "desc",
) -> dict:
    dirs = [p for p in diary_dirs(diary_root) if in_date_range(p.name, start_date, end_date)]
    dirs = sorted(dirs, key=lambda p: p.name, reverse=(order != "asc"))
    results = []
    for diary_dir in dirs[: max(1, min(int(limit), 200))]:
        diary_path = diary_dir / diary_name
        comment_path = diary_dir / comment_name
        results.append({
            "diary_id": diary_dir.name,
            "date": diary_dir.name[:10],
            "time": diary_dir.name[11:].replace("-", ":"),
            "title": title_for(diary_dir, title_name),
            "has_diary": diary_path.exists(),
            "has_comment": comment_path.exists(),
            "diary_line_count": line_count(diary_path),
            "comment_line_count": line_count(comment_path),
        })
    return {"results": results}


def search_title(
    diary_root: Path,
    title_name: str,
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> dict:
    needle = query.casefold()
    results = []
    for diary_dir in diary_dirs(diary_root):
        if not in_date_range(diary_dir.name, start_date, end_date):
            continue
        title = title_for(diary_dir, title_name)
        if title and needle in title.casefold():
            results.append({
                "diary_id": diary_dir.name,
                "date": diary_dir.name[:10],
                "title": title,
            })
            if len(results) >= max(1, min(int(limit), 100)):
                break
    return {"results": results}


def search_keyword(
    diary_root: Path,
    diary_name: str,
    comment_name: str,
    query: str,
    diary_or_comment: str = "diary",
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> dict:
    if not query:
        return {"error": "query must not be empty", "results": []}
    needle = query.casefold()
    results = []
    max_results = max(1, min(int(limit), 100))
    for diary_dir in diary_dirs(diary_root):
        if not in_date_range(diary_dir.name, start_date, end_date):
            continue
        path = entry_path(diary_root, diary_dir.name, diary_name, comment_name, diary_or_comment)
        if not path.exists():
            continue
        for line_number, line in enumerate(read_lines(path), start=1):
            start = line.casefold().find(needle)
            if start == -1:
                continue
            results.append(format_line_result(diary_dir.name, diary_or_comment, line_number, line, start, start + len(query)))
            if len(results) >= max_results:
                return {"results": results}
    return {"results": results}


def search_regex(
    diary_root: Path,
    diary_name: str,
    comment_name: str,
    pattern: str,
    diary_or_comment: str = "diary",
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> dict:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"error": f"Invalid regex: {e}", "results": []}

    results = []
    max_results = max(1, min(int(limit), 100))
    for diary_dir in diary_dirs(diary_root):
        if not in_date_range(diary_dir.name, start_date, end_date):
            continue
        path = entry_path(diary_root, diary_dir.name, diary_name, comment_name, diary_or_comment)
        if not path.exists():
            continue
        for line_number, line in enumerate(read_lines(path), start=1):
            match = regex.search(line)
            if not match:
                continue
            results.append(format_line_result(diary_dir.name, diary_or_comment, line_number, line, match.start(), match.end()))
            if len(results) >= max_results:
                return {"results": results}
    return {"results": results}


def read_entry_lines(
    diary_root: Path,
    diary_name: str,
    comment_name: str,
    diary_id: str,
    start_line: int,
    end_line: int,
    diary_or_comment: str = "diary",
) -> dict:
    path = entry_path(diary_root, diary_id, diary_name, comment_name, diary_or_comment)
    if not path.exists():
        return {"error": f"Missing {diary_or_comment}: {path}"}
    lines = read_lines(path)
    if not lines:
        return {
            "diary_id": diary_id,
            "diary_or_comment": diary_or_comment,
            "line_start": 0,
            "line_end": 0,
            "lines": [],
        }
    start = max(1, int(start_line))
    end = min(len(lines), int(end_line))
    if start > end:
        return {"error": f"Invalid line range: {start_line}-{end_line}", "line_count": len(lines)}
    selected = [
        {"line": line_number, "text": lines[line_number - 1]}
        for line_number in range(start, end + 1)
    ]
    return {
        "diary_id": diary_id,
        "date": diary_id[:10],
        "diary_or_comment": diary_or_comment,
        "line_start": start,
        "line_end": end,
        "ref": make_ref(diary_id, diary_or_comment, start, end),
        "lines": selected,
    }


def run_research_tool_call(tool_call: dict, diary_root: Path, diary_name: str, comment_name: str, title_name: str) -> dict:
    function = tool_call.get("function") or {}
    try:
        name = function.get("name", "")
        args = json.loads(function.get("arguments") or "{}")
        if name == "list_diaries":
            return list_diaries(
                diary_root,
                diary_name,
                comment_name,
                title_name,
                args.get("start_date"),
                args.get("end_date"),
                int(args.get("limit", 30)),
                str(args.get("order", "desc")),
            )
        if name == "search_title":
            return search_title(
                diary_root,
                title_name,
                str(args["query"]),
                args.get("start_date"),
                args.get("end_date"),
                int(args.get("limit", 20)),
            )
        if name == "search_keyword":
            return search_keyword(
                diary_root,
                diary_name,
                comment_name,
                str(args["query"]),
                str(args.get("diary_or_comment", "diary")),
                args.get("start_date"),
                args.get("end_date"),
                int(args.get("limit", 20)),
            )
        if name == "search_regex":
            return search_regex(
                diary_root,
                diary_name,
                comment_name,
                str(args["pattern"]),
                str(args.get("diary_or_comment", "diary")),
                args.get("start_date"),
                args.get("end_date"),
                int(args.get("limit", 20)),
            )
        if name == "read_entry_lines":
            return read_entry_lines(
                diary_root,
                diary_name,
                comment_name,
                str(args["diary_id"]),
                int(args["start_line"]),
                int(args["end_line"]),
                str(args.get("diary_or_comment", "diary")),
            )
        return {"error": f"Unknown tool: {name}", "got": args}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid tool arguments JSON: {e}", "got": function.get("arguments")}
    except Exception as e:
        return {"error": f"Tool call failed: {type(e).__name__}: {e}", "got": function.get("arguments")}

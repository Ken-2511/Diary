import json
import re
from pathlib import Path


DIARY_ID_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")
VALID_SOURCES = {"diary", "comment"}
PUNCTUATION = "\u3002\uff01\uff1f!?;,\uff1b\uff0c"
DEFAULT_SEARCH_LIMIT = 200
MAX_SEARCH_LIMIT = 500

MAIN_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "ask_memory",
            "description": "Ask a memory research sub-agent a natural-language question about past diaries. The sub-agent searches diary/comment lines and returns a precise answer with evidence quotes.",
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
            "description": "List past diary folders with metadata. Use date filters to inspect recent or historical periods before searching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": ["string", "null"], "default": None},
                    "end_date": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": DEFAULT_SEARCH_LIMIT},
                    "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_title",
            "description": "Search past diary titles by case-insensitive literal text. Returns total match counts and truncation metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "start_date": {"type": ["string", "null"], "default": None},
                    "end_date": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": DEFAULT_SEARCH_LIMIT},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search past diary or comment lines. Prefer mode='regex' for combined or narrowed searches; use mode='literal' only for exact plain text. Returns total match counts, truncation metadata, short quotes, full lines, and refs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {"type": "string", "enum": ["regex", "literal"], "default": "regex"},
                    "diary_or_comment": {"type": "string", "enum": ["diary", "comment"], "default": "diary"},
                    "start_date": {"type": ["string", "null"], "default": None},
                    "end_date": {"type": ["string", "null"], "default": None},
                    "limit": {"type": "integer", "default": DEFAULT_SEARCH_LIMIT},
                },
                "required": ["query"],
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


def in_scope(diary_id: str, start_date: str | None, end_date: str | None, target_id: str | None = None) -> bool:
    date = diary_id[:10]
    if target_id and diary_id >= target_id:
        return False
    return (start_date is None or date >= start_date) and (end_date is None or date <= end_date)


def normalized_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_SEARCH_LIMIT))


def result_meta(total: int, returned: int, limit: int, target_id: str | None) -> dict:
    truncated = total > returned
    meta = {
        "total_matches": total,
        "returned_count": returned,
        "limit": limit,
        "truncated": truncated,
        "scope": "past_diaries_only" if target_id else "all_diaries",
    }
    if truncated:
        meta["suggestion"] = (
            f"{total} matches found; only {returned} returned. "
            "Narrow the search with start_date/end_date or a more specific regex."
        )
    return meta


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
    limit: int = DEFAULT_SEARCH_LIMIT,
    order: str = "desc",
    target_id: str | None = None,
) -> dict:
    dirs = [p for p in diary_dirs(diary_root) if in_scope(p.name, start_date, end_date, target_id)]
    dirs = sorted(dirs, key=lambda p: p.name, reverse=(order != "asc"))
    max_results = normalized_limit(limit)
    results = []
    for diary_dir in dirs[:max_results]:
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
    return result_meta(len(dirs), len(results), max_results, target_id) | {"results": results}


def search_title(
    diary_root: Path,
    title_name: str,
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
    target_id: str | None = None,
) -> dict:
    if not query:
        return {"error": "query must not be empty", "results": []}
    needle = query.casefold()
    max_results = normalized_limit(limit)
    results = []
    total = 0
    for diary_dir in diary_dirs(diary_root):
        if not in_scope(diary_dir.name, start_date, end_date, target_id):
            continue
        title = title_for(diary_dir, title_name)
        if title and needle in title.casefold():
            total += 1
            if len(results) < max_results:
                results.append({
                    "diary_id": diary_dir.name,
                    "date": diary_dir.name[:10],
                    "title": title,
                })
    return result_meta(total, len(results), max_results, target_id) | {"results": results}


def compile_text_query(query: str, mode: str) -> tuple[re.Pattern | None, str | None]:
    if not query:
        return None, "query must not be empty"
    if mode == "literal":
        query = re.escape(query)
    elif mode != "regex":
        return None, f"Invalid search mode: {mode}"
    try:
        return re.compile(query, re.IGNORECASE), None
    except re.error as e:
        return None, f"Invalid regex: {e}"


def search_text(
    diary_root: Path,
    diary_name: str,
    comment_name: str,
    query: str,
    mode: str = "regex",
    diary_or_comment: str = "diary",
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
    target_id: str | None = None,
) -> dict:
    regex, error = compile_text_query(query, mode)
    if error:
        return {"error": error, "results": []}

    results = []
    total = 0
    max_results = normalized_limit(limit)
    for diary_dir in diary_dirs(diary_root):
        if not in_scope(diary_dir.name, start_date, end_date, target_id):
            continue
        path = entry_path(diary_root, diary_dir.name, diary_name, comment_name, diary_or_comment)
        if not path.exists():
            continue
        for line_number, line in enumerate(read_lines(path), start=1):
            match = regex.search(line)
            if not match:
                continue
            total += 1
            if len(results) < max_results:
                results.append(format_line_result(diary_dir.name, diary_or_comment, line_number, line, match.start(), match.end()))

    return result_meta(total, len(results), max_results, target_id) | {
        "mode": mode,
        "diary_or_comment": diary_or_comment,
        "results": results,
    }


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


def run_research_tool_call(tool_call: dict, diary_root: Path, diary_name: str, comment_name: str, title_name: str, target_id: str | None) -> dict:
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
                int(args.get("limit", DEFAULT_SEARCH_LIMIT)),
                str(args.get("order", "desc")),
                target_id,
            )
        if name == "search_title":
            return search_title(
                diary_root,
                title_name,
                str(args["query"]),
                args.get("start_date"),
                args.get("end_date"),
                int(args.get("limit", DEFAULT_SEARCH_LIMIT)),
                target_id,
            )
        if name == "search_text":
            return search_text(
                diary_root,
                diary_name,
                comment_name,
                str(args["query"]),
                str(args.get("mode", "regex")),
                str(args.get("diary_or_comment", "diary")),
                args.get("start_date"),
                args.get("end_date"),
                int(args.get("limit", DEFAULT_SEARCH_LIMIT)),
                target_id,
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

import argparse
import io
import json
import os
import sys
import tomllib
from pathlib import Path

import requests

from agent_tools import (
    DIARY_ID_RE,
    MAIN_SCHEMAS,
    RESEARCH_SCHEMAS,
    diary_dirs,
    read_entry_lines,
    run_research_tool_call,
)


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG = tomllib.loads((CONFIG_DIR / "config.toml").read_text(encoding="utf-8"))
DIARY_NAME = CONFIG.get("diary_name", "diary.txt")
TITLE_NAME = CONFIG.get("title_name", "title.txt")
COMMENT_NAME = CONFIG.get("comment_name", "comment.txt")
TEMP_DIR = BASE_DIR / "temp"
COMMENT_PATH = TEMP_DIR / "comment.txt"
CANDIDATE_PATH = TEMP_DIR / "comment_candidate.txt"
SUMMARY_PATH = TEMP_DIR / "comment_run_summary.json"
CONTEXT_DIR = TEMP_DIR / "comment_contexts"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def cfg_int(name: str, default: int) -> int:
    return int(CONFIG.get(name, default))


CHAT_MODEL = CONFIG.get("model", "moonshotai/kimi-k2.6")
MEMORY_MODEL = CONFIG.get("memory_model", CHAT_MODEL)
MAX_MAIN_STEPS = cfg_int("max_main_steps", 25)
SUGGESTED_MEMORY_QUESTIONS = cfg_int("suggested_memory_questions", 8)
MAX_MEMORY_QUESTIONS = cfg_int("max_memory_questions", 15)
MAX_RESEARCH_TOOL_CALLS = cfg_int("max_research_tool_calls_per_question", 20)


def read_prompt(name: str) -> str:
    return (CONFIG_DIR / name).read_text(encoding="utf-8").strip()


def headers() -> dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    result = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if site := os.environ.get("OPENROUTER_SITE_URL"):
        result["HTTP-Referer"] = site
    if title := os.environ.get("OPENROUTER_APP_NAME"):
        result["X-Title"] = title
    return result


def chat_stream(messages: list[dict], tools: list[dict], model: str, label: str, print_content: bool = True) -> dict:
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    api_messages = [{k: v for k, v in message.items() if k in allowed} for message in messages]
    payload = {
        "model": model,
        "messages": api_messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "reasoning": {"enabled": True, "effort": "medium", "exclude": True},
    }

    response = requests.post(CHAT_URL, headers=headers(), json=payload, stream=True, timeout=600)
    response.raise_for_status()
    response.encoding = "utf-8"

    content, buffer, tool_calls, done = [], "", {}, False
    for piece in response.iter_content(4096, decode_unicode=True):
        buffer += piece
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            data = "\n".join(line[5:].strip() for line in event.split("\n") if line.startswith("data:"))
            if not data:
                continue
            if data == "[DONE]":
                done = True
                break
            event_json = json.loads(data)
            delta = (event_json.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                if print_content:
                    print(delta["content"], end="", flush=True)
                content.append(delta["content"])
            for tool_call in delta.get("tool_calls") or []:
                index = tool_call.get("index", len(tool_calls))
                saved = tool_calls.setdefault(index, {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tool_call.get("id"):
                    saved["id"] += tool_call["id"]
                if tool_call.get("type"):
                    saved["type"] = tool_call["type"]
                function = tool_call.get("function") or {}
                if function.get("name"):
                    saved["function"]["name"] += function["name"]
                if function.get("arguments"):
                    saved["function"]["arguments"] += function["arguments"]
        if done:
            break
    if print_content:
        print()
    return {
        "label": label,
        "content": "".join(content).strip(),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
    }


def target_payload(target: Path) -> tuple[str, str]:
    title = (target / TITLE_NAME).read_text(encoding="utf-8").strip() if (target / TITLE_NAME).exists() else ""
    diary = (target / DIARY_NAME).read_text(encoding="utf-8").strip()
    content = (
        f"Target diary id:\n{target.name}\n\n"
        f"Target date:\n{target.name[:10]}\n\n"
        f"Target time:\n{target.name[11:].replace('-', ':')}\n\n"
        f"Target title:\n{title}\n\n"
        f"Suggested memory questions: about {SUGGESTED_MEMORY_QUESTIONS}\n"
        f"Maximum memory questions: {MAX_MEMORY_QUESTIONS}\n\n"
        f"Target diary:\n{diary}"
    )
    return title, content


def summarize_tool_call(tool_call: dict, result: dict) -> dict:
    function = tool_call.get("function") or {}
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {"raw": function.get("arguments")}
    summary = {
        "name": function.get("name", ""),
        "arguments": args,
    }
    if "results" in result:
        summary["result_count"] = len(result.get("results") or [])
        if "total_matches" in result:
            summary["total_matches"] = result["total_matches"]
        if "returned_count" in result:
            summary["returned_count"] = result["returned_count"]
        if "truncated" in result:
            summary["truncated"] = result["truncated"]
        if "suggestion" in result:
            summary["suggestion"] = result["suggestion"]
        summary["refs"] = [item.get("ref") for item in (result.get("results") or [])[:5] if item.get("ref")]
    elif result.get("ref"):
        summary["refs"] = [result["ref"]]
    if result.get("error"):
        summary["error"] = result["error"]
    return summary


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def prepare_context_dir(target_id: str) -> Path:
    context_dir = CONTEXT_DIR / target_id
    context_dir.mkdir(parents=True, exist_ok=True)
    for old_path in context_dir.glob("*.json"):
        old_path.unlink()
    return context_dir


def save_memory_context(
    context_dir: Path,
    memory_index: int,
    target: Path,
    question: str,
    state: list[dict],
    answer: str,
    tool_summaries: list[dict],
    tool_count: int,
    error: str | None = None,
) -> str:
    payload = {
        "agent": "memory",
        "memory_index": memory_index,
        "target_id": target.name,
        "question": question,
        "model": MEMORY_MODEL,
        "max_research_tool_calls": MAX_RESEARCH_TOOL_CALLS,
        "messages": state,
        "final_answer": answer,
        "tool_calls": tool_summaries,
        "tool_call_count": tool_count,
    }
    if error:
        payload["error"] = error
    return write_json(context_dir / f"memory_{memory_index:02d}.json", payload)


def save_main_context(
    context_dir: Path,
    target: Path,
    state: list[dict],
    final_comment: str,
    main_tool_calls: list[dict],
    memory_questions: list[dict],
) -> str:
    return write_json(context_dir / "main.json", {
        "agent": "main",
        "target_id": target.name,
        "model": CHAT_MODEL,
        "max_main_steps": MAX_MAIN_STEPS,
        "messages": state,
        "final_comment": final_comment,
        "main_tool_calls": main_tool_calls,
        "memory_context_paths": [
            item["context_path"]
            for item in memory_questions
            if item.get("context_path")
        ],
    })


def run_memory_agent(question: str, target: Path, diary_root: Path, context_dir: Path, memory_index: int) -> dict:
    _, target_context = target_payload(target)
    state = [
        {"role": "system", "content": read_prompt("memory_system.prompt.md")},
        {
            "role": "user",
            "content": (
                f"Memory question:\n{question}\n\n"
                "You are researching background for a new comment. Do not read the target diary's existing comment "
                "unless the question explicitly asks about comments or previous AI feedback.\n\n"
                f"{target_context}"
            ),
        },
    ]
    tool_summaries = []
    tool_count = 0
    budget_answer_requested = False

    for step_num in range(1, MAX_RESEARCH_TOOL_CALLS + 4):
        call = chat_stream(state, RESEARCH_SCHEMAS, MEMORY_MODEL, f"memory_{step_num}", print_content=False)
        assistant_message = {"role": "assistant", "content": call["content"] or None}
        if call["tool_calls"]:
            assistant_message["tool_calls"] = call["tool_calls"]
        state.append(assistant_message)
        if not call["tool_calls"]:
            if not call["content"]:
                state.append({
                    "role": "user",
                    "content": "Empty response received. Continue researching with tools, or provide the final structured memory answer if research is complete.",
                })
                continue
            context_path = save_memory_context(context_dir, memory_index, target, question, state, call["content"], tool_summaries, tool_count)
            return {
                "question": question,
                "answer": call["content"],
                "tool_calls": tool_summaries,
                "tool_call_count": tool_count,
                "context_path": context_path,
            }

        budget_exceeded = False
        for tool_call in call["tool_calls"]:
            tool_count += 1
            if tool_count > MAX_RESEARCH_TOOL_CALLS:
                budget_exceeded = True
                result = {
                    "error": f"Research tool budget exceeded: {MAX_RESEARCH_TOOL_CALLS}",
                    "instruction": "Do not call more tools. Answer the memory question now using the evidence already returned in this context.",
                }
            else:
                result = run_research_tool_call(tool_call, diary_root, DIARY_NAME, COMMENT_NAME, TITLE_NAME, target.name)
            tool_summaries.append(summarize_tool_call(tool_call, result))
            state.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False),
            })

        if budget_exceeded:
            if budget_answer_requested:
                answer = "Research stopped because the tool-call budget was exceeded."
                context_path = save_memory_context(
                    context_dir,
                    memory_index,
                    target,
                    question,
                    state,
                    answer,
                    tool_summaries,
                    tool_count,
                    f"Research tool budget exceeded: {MAX_RESEARCH_TOOL_CALLS}",
                )
                return {
                    "question": question,
                    "answer": answer,
                    "tool_calls": tool_summaries,
                    "tool_call_count": tool_count,
                    "error": f"Research tool budget exceeded: {MAX_RESEARCH_TOOL_CALLS}",
                    "context_path": context_path,
                }
            budget_answer_requested = True
            state.append({
                "role": "user",
                "content": (
                    f"You have reached the research tool budget ({MAX_RESEARCH_TOOL_CALLS}). "
                    "Do not call more tools. Answer the memory question now using only the evidence already returned above. "
                    "Use the required Answer/Evidence format."
                ),
            })

    context_path = save_memory_context(
        context_dir,
        memory_index,
        target,
        question,
        state,
        "Research did not finish.",
        tool_summaries,
        tool_count,
        "Research did not finish.",
    )
    return {
        "question": question,
        "answer": "Research did not finish.",
        "tool_calls": tool_summaries,
        "tool_call_count": tool_count,
        "error": "Research did not finish.",
        "context_path": context_path,
    }


def run_main_tool_call(tool_call: dict, target: Path, diary_root: Path, memory_questions: list[dict], context_dir: Path) -> dict:
    function = tool_call.get("function") or {}
    try:
        name = function.get("name", "")
        args = json.loads(function.get("arguments") or "{}")
        if name == "ask_memory":
            if len(memory_questions) >= MAX_MEMORY_QUESTIONS:
                return {"error": f"Memory question budget exceeded: {MAX_MEMORY_QUESTIONS}"}
            result = run_memory_agent(str(args["question"]), target, diary_root, context_dir, len(memory_questions) + 1)
            memory_questions.append(result)
            return {
                "question": result["question"],
                "answer": result["answer"],
                "tool_call_count": result["tool_call_count"],
                "error": result.get("error"),
                "context_path": result.get("context_path"),
            }
        if name == "read_entry_lines":
            return read_entry_lines(
                diary_root,
                DIARY_NAME,
                COMMENT_NAME,
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


def select_target(diary_root: Path, target_id: str | None, no_save: bool) -> Path:
    if target_id:
        if not DIARY_ID_RE.fullmatch(target_id):
            raise RuntimeError(f"Invalid target id: {target_id}")
        target = diary_root / target_id
        if not target.exists():
            raise RuntimeError(f"Target diary folder not found: {target}")
        if (target / COMMENT_NAME).exists() and not no_save:
            raise RuntimeError("Target already has a comment. Use --no-save to avoid overwriting it.")
        return target

    targets = [p for p in diary_dirs(diary_root) if (p / DIARY_NAME).exists() and not (p / COMMENT_NAME).exists()]
    if not targets:
        raise RuntimeError("No diary without comment found")
    return targets[0]


def save_summary(summary: dict) -> None:
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a diary comment with a line-first memory agent.")
    parser.add_argument("--diary-root", default=CONFIG["diary_dir"])
    parser.add_argument("--target-id")
    parser.add_argument("--no-save", action="store_true", help="Write only a temp candidate and do not overwrite the target comment.")
    args = parser.parse_args()

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    TEMP_DIR.mkdir(exist_ok=True)
    diary_root = Path(args.diary_root)
    target = select_target(diary_root, args.target_id, args.no_save)
    context_dir = prepare_context_dir(target.name)
    if not (target / DIARY_NAME).read_text(encoding="utf-8").strip():
        print(f"Diary is empty: {target / DIARY_NAME}")
        print("Please write something in the diary before generating a comment.")
        return

    _, user_content = target_payload(target)
    state = [
        {"role": "system", "content": read_prompt("agent_system.prompt.md")},
        {"role": "user", "content": user_content},
    ]
    memory_questions: list[dict] = []
    main_tool_calls = []
    final_comment = ""

    for step_num in range(1, MAX_MAIN_STEPS + 1):
        print(f"\nMain agent step {step_num}...")
        call = chat_stream(state, MAIN_SCHEMAS, CHAT_MODEL, f"main_{step_num}")
        assistant_message = {"role": "assistant", "content": call["content"] or None}
        if call["tool_calls"]:
            assistant_message["tool_calls"] = call["tool_calls"]
        state.append(assistant_message)
        if not call["tool_calls"]:
            if not call["content"]:
                state.append({
                    "role": "user",
                    "content": "Empty response received. Continue with a tool call, or write the final diary comment if you have enough context.",
                })
                continue
            final_comment = call["content"]
            break
        for tool_call in call["tool_calls"]:
            result = run_main_tool_call(tool_call, target, diary_root, memory_questions, context_dir)
            main_tool_calls.append(summarize_tool_call(tool_call, result))
            state.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False),
            })

    if not final_comment:
        raise RuntimeError(f"Main agent did not finish within {MAX_MAIN_STEPS} steps")

    temp_output = CANDIDATE_PATH if args.no_save else COMMENT_PATH
    temp_output.write_text(final_comment, encoding="utf-8")
    saved_paths = [str(temp_output)]
    if not args.no_save:
        target_output = target / COMMENT_NAME
        target_output.write_text(final_comment, encoding="utf-8")
        saved_paths.append(str(target_output))
    main_context_path = save_main_context(context_dir, target, state, final_comment, main_tool_calls, memory_questions)

    summary = {
        "target_id": target.name,
        "target_path": str(target),
        "chat_model": CHAT_MODEL,
        "memory_model": MEMORY_MODEL,
        "max_memory_questions": MAX_MEMORY_QUESTIONS,
        "max_research_tool_calls_per_question": MAX_RESEARCH_TOOL_CALLS,
        "memory_question_count": len(memory_questions),
        "memory_questions": memory_questions,
        "main_tool_calls": main_tool_calls,
        "context_dir": str(context_dir),
        "main_context_path": main_context_path,
        "memory_context_paths": [
            item["context_path"]
            for item in memory_questions
            if item.get("context_path")
        ],
        "comment_char_count": len(final_comment),
        "no_save": args.no_save,
        "saved_paths": saved_paths,
    }
    save_summary(summary)
    for path in saved_paths:
        print(f"saved: {path}")
    print(f"saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

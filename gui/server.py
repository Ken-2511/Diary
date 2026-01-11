"""
FastAPI 后端服务器
启动时加载所有日记到内存缓存
"""

import os
import re
import sys
import time
import threading
import tomllib
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 加载配置
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.toml")
with open(CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)

DIARY_DIR = config["diary_dir"]
DIARY_NAME = config["diary_name"]
COMMENT_NAME = config["comment_name"]
TITLE_NAME = config["title_name"]
USAGE_NAME = config.get("usage_name", "usage.json")

# 日记缓存
DIARY_CACHE = {}

# 心跳相关
LAST_HEARTBEAT = time.time()
HEARTBEAT_TIMEOUT = 10  # 秒，超过这个时间没有心跳就退出
STARTUP_GRACE_PERIOD = 5  # 启动后的宽限期


def check_heartbeat():
    """后台线程：检查心跳超时"""
    global LAST_HEARTBEAT
    # 启动后等待一段时间，给浏览器打开的时间
    time.sleep(STARTUP_GRACE_PERIOD)
    LAST_HEARTBEAT = time.time()  # 重置心跳时间
    
    while True:
        time.sleep(2)
        if time.time() - LAST_HEARTBEAT > HEARTBEAT_TIMEOUT:
            print("\n[Server] No heartbeat received, shutting down...")
            os._exit(0)


def load_file_content(path: str) -> Optional[str]:
    """安全读取文件内容"""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def load_usage(path: str) -> Optional[dict]:
    """读取 usage.json"""
    import json
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def parse_date(dir_name: str) -> datetime:
    """解析目录名为 datetime"""
    return datetime.strptime(dir_name, "%Y-%m-%d-%H-%M-%S")


def format_date_display(dir_name: str) -> str:
    """格式化日期用于显示"""
    dt = parse_date(dir_name)
    return dt.strftime("%Y-%m-%d %H:%M")


def load_all_diaries():
    """加载所有日记到缓存"""
    global DIARY_CACHE
    DIARY_CACHE = {}
    
    dir_names = os.listdir(DIARY_DIR)
    dir_names = [name for name in dir_names if re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", name)]
    dir_names.sort(key=lambda x: x.split("-"), reverse=True)  # 倒序，最新的在前
    
    for dir_name in dir_names:
        dir_path = os.path.join(DIARY_DIR, dir_name)
        
        diary_content = load_file_content(os.path.join(dir_path, DIARY_NAME))
        comment_content = load_file_content(os.path.join(dir_path, COMMENT_NAME))
        title_content = load_file_content(os.path.join(dir_path, TITLE_NAME))
        usage = load_usage(os.path.join(dir_path, USAGE_NAME))
        
        DIARY_CACHE[dir_name] = {
            "date": dir_name,
            "date_display": format_date_display(dir_name),
            "title": title_content.strip() if title_content else "无标题",
            "diary": diary_content or "",
            "comment": comment_content,
            "has_comment": comment_content is not None,
            "usage": usage,
            "word_count": len(diary_content) if diary_content else 0,
        }
    
    print(f"Loaded {len(DIARY_CACHE)} diaries into cache")


# 创建 FastAPI 应用
app = FastAPI(title="Diary GUI")


@app.on_event("startup")
async def startup_event():
    """服务启动时加载所有日记，并启动心跳检测线程"""
    load_all_diaries()
    # 启动心跳检测线程
    heartbeat_thread = threading.Thread(target=check_heartbeat, daemon=True)
    heartbeat_thread.start()


@app.get("/api/heartbeat")
async def heartbeat():
    """接收前端心跳"""
    global LAST_HEARTBEAT
    LAST_HEARTBEAT = time.time()
    return {"status": "ok"}


# API 路由
@app.get("/api/diaries")
async def get_diaries():
    """获取所有日记的元信息"""
    return [
        {
            "date": d["date"],
            "date_display": d["date_display"],
            "title": d["title"],
            "has_comment": d["has_comment"],
            "word_count": d["word_count"],
        }
        for d in DIARY_CACHE.values()
    ]


@app.get("/api/diary/{date}")
async def get_diary(date: str):
    """获取指定日期的日记详情"""
    if date in DIARY_CACHE:
        return DIARY_CACHE[date]
    return {"error": "Diary not found"}


@app.get("/api/search")
async def search_diaries(q: str = Query(..., min_length=1)):
    """搜索日记内容"""
    q_lower = q.lower()
    results = []
    
    for d in DIARY_CACHE.values():
        # 搜索标题、日记内容、评论
        title_match = q_lower in d["title"].lower()
        diary_match = q_lower in d["diary"].lower()
        comment_match = d["comment"] and q_lower in d["comment"].lower()
        
        if title_match or diary_match or comment_match:
            # 提取匹配片段作为预览
            preview = ""
            if diary_match:
                idx = d["diary"].lower().find(q_lower)
                start = max(0, idx - 30)
                end = min(len(d["diary"]), idx + len(q) + 30)
                preview = "..." + d["diary"][start:end] + "..."
            
            results.append({
                "date": d["date"],
                "date_display": d["date_display"],
                "title": d["title"],
                "has_comment": d["has_comment"],
                "preview": preview,
                "match_in": {
                    "title": title_match,
                    "diary": diary_match,
                    "comment": comment_match,
                }
            })
    
    return results


@app.get("/api/stats")
async def get_stats():
    """获取统计信息"""
    total = len(DIARY_CACHE)
    with_comment = sum(1 for d in DIARY_CACHE.values() if d["has_comment"])
    without_comment = total - with_comment
    total_words = sum(d["word_count"] for d in DIARY_CACHE.values())
    
    # 本月统计
    now = datetime.now()
    this_month = sum(
        1 for d in DIARY_CACHE.values()
        if parse_date(d["date"]).year == now.year and parse_date(d["date"]).month == now.month
    )
    
    return {
        "total": total,
        "with_comment": with_comment,
        "without_comment": without_comment,
        "total_words": total_words,
        "this_month": this_month,
    }


@app.post("/api/reload")
async def reload_diaries():
    """重新加载日记缓存"""
    load_all_diaries()
    return {"message": f"Reloaded {len(DIARY_CACHE)} diaries"}


# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    """返回主页"""
    return FileResponse("static/index.html")

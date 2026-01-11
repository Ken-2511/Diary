"""
GUI 入口文件
启动 FastAPI 后端并自动打开浏览器
"""

import webbrowser
import threading
import time
import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def open_browser():
    """等待服务器启动后打开浏览器"""
    time.sleep(1.5)  # 等待服务器启动
    webbrowser.open(f"http://{HOST}:{PORT}")


def main():
    # 在后台线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动 FastAPI 服务器
    print(f"Starting server at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the server")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    import os
    # 切换到 gui 目录，确保相对路径正确
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

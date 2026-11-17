"""config.py — 配置集中（L4）：所有"从环境读东西"只住这一个文件

【学徒注】之前 .env 的加载散在三处（agent.py / web.py / http_entry.py
各写一遍 load_dotenv），路径常量也各算各的——改一处忘两处的温床。
集中后：路径、角色、传输这些"开机关"都从这里出；缺关键钥匙时
fail fast（把缺的钥匙一次报全，别一把一把挤牙膏）。

真 .env 不进库（密钥）；.env.example 进库当模板——新人 copy 一份
改钥匙就能起。两把钥匙的分工见 web.py 模块注释：WEB_API_TOKEN 管
外网门（浏览器→web），MCP_API_KEY 管内网门（web→MCP Server）。
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # 进程里第一次 import config 就把 .env 灌进来

# ---------- 路径 ----------
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
SESSIONS_FILE = DATA_DIR / "sessions.json"  # Web 会话账（运行时账，gitignore）

# ---------- 开机关（非密钥，环境变量直接给） ----------
APP_ROLE = os.environ.get("APP_ROLE", "trader").strip() or "trader"
# MCP_TRANSPORT 原样给出（解析逻辑住 agent.py 的 _http_servers——
# 它是"挂载配置怎么写"的事，不是纯配置）
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "")


def require_secret(*names: str) -> None:
    """缺关键钥匙就当场退出（fail fast），把缺的一次报全"""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"[配置错误] .env 缺少：{', '.join(missing)}（模板见 .env.example）")

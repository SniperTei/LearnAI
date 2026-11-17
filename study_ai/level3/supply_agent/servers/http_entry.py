"""http_entry.py — 六座 Server 共用的 --http 启动入口（L2）

【学徒注】stdio 和 HTTP 两种模式的分工：
  - stdio（默认）：agent.py 当子进程拉起，stdout 是协议通道——安静是纪律
  - --http：自己常驻运行，谁都能连（本机学习版 = 127.0.0.1 + 静态 Bearer key）

认证：fastmcp 4.2 没有现成的"静态 token"认证器（TokenVerifier 是基类），
自己 subclass 一个——verify_token 对上了就发 AccessToken，对不上返回 None
（框架转成 401）。AccessToken 的 scopes 是必填字段（冒烟实验踩的坑，漏了
直接 500）。

/health：fastmcp 4.2 也没内置（实测 404）。mcp.http_app() 返回的是
Starlette app，路由表可以自己加——挂在最后，不跟 /mcp 抢道。

端口分工（与 agent.py 的 SERVER_PORTS 必须一致，改要两边改）：
  inventory 8101 / supplier 8102 / logistics 8103 /
  procurement 8104 / customer 8105 / sales 8106
环境变量可覆盖：MCP_HOST（默认 127.0.0.1，L5 Docker 要 0.0.0.0）、
  <SERVER>_PORT（如 INVENTORY_PORT）。
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import uvicorn
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from fastmcp.server.auth import AccessToken, TokenVerifier

# config.py 住项目根，Server 子进程的 sys.path 只有 servers/——先把根插进来
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402（路径调整必须先于 import）


class StaticTokenVerifier(TokenVerifier):
    """静态 Bearer token 认证（学习版：一把钥匙对一个客户端）

    【学徒注】真实系统该换 JWT/OAuth（钥匙能过期、能吊销、能区分身份），
    静态 key 的问题：泄露了只能换一把，所有客户端同时重配。
    """

    def __init__(self, token: str):
        super().__init__()
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        if token == self._token:
            return AccessToken(token=token, client_id="supply-agent", scopes=[])
        return None  # 返回 None → 框架回 401


def run_http(mcp, default_port: int) -> None:
    """--http 入口共用体：验钥匙 → 挂认证 → 加 /health → 开日志 → 常驻运行"""
    config.require_secret("MCP_API_KEY")  # L4：fail fast 集中进 config

    mcp.auth = StaticTokenVerifier(os.environ["MCP_API_KEY"])
    app = mcp.http_app()
    app.router.routes.append(
        Route("/health", lambda request: PlainTextResponse("ok")))
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get(f"{mcp.name.upper()}_PORT", str(default_port)))

    # L4：uvicorn 的访问日志（谁来了、什么状态、多久）落进 logs/mcp_<名>.log，
    # JSON 行格式与 web.log 同款——排查时两边日志一个 grep 通吃
    config.LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(config.LOG_DIR / f"mcp_{mcp.name}.log",
                                  encoding="utf-8")

    class _JsonLine(logging.Formatter):
        def format(self, record):
            return json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"),
                               "server": mcp.name,
                               "msg": record.getMessage()},
                              ensure_ascii=False)

    handler.setFormatter(_JsonLine())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    print(f"[{mcp.name}] HTTP 模式 http://{host}:{port}/mcp（/health 可探活，"
          f"日志 logs/mcp_{mcp.name}.log）", file=sys.stderr)
    # log_config=None：别让 uvicorn 重配 logging 把我们的 handler 顶掉
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)

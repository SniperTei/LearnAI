"""web.py — FastAPI Web 接口（L3）：CLI 之外的第二个门脸

架构：
  static/index.html（浏览器门脸）
    └── fetch /api/*（Bearer WEB_API_TOKEN 认证）
          └── web.py（本文件，FastAPI）
                └── agent.build_agent() / agent.run_turn()（L3 抽的公共件）
                      └── MCPAdapter → 六座 MCP Server（照旧，一行不改）

【学徒注】与 FoodRAG 的对照（当时的历史方案 vs 现在的会话方案）：
  FoodRAG：服务端零状态，history 由前端持有、每次请求带回——
            服务器简单，但刷新页面前端得自己保管好账
  本项目：会话住服务端 data/sessions.json，前端只报 session_id——
            刷新/换设备都不丢，代价是服务器多管一本账（生命周期与
            业务账本不同，所以单独一个文件，不混进 data/*.json 业务账）

教学要点：
1. 两把钥匙分开：WEB_API_TOKEN（外网门，浏览器→web）与 MCP_API_KEY
   （内网门，web→MCP Server）不是一把——外网钥匙泄露换外网的，
   不用动内网。最小权限原则的粗粒度版。
2. 金鱼记忆：喂 agent 的历史只取最近 2 轮纯文本（user/assistant 成对）——
   确认闸门要跨一轮（下单复述→"确认"），2 轮刚好够；全量喂会把
   工具轨迹也滚进去，token 越滚越贵（CLI 是终端学习场景才全量喂）。
3. sessions.json 同样守"读一份→改这份→存这份"纪律；并发对话用一把
   asyncio 锁串行化——学习版的答案，真药方（按会话加锁/DB行锁）
   记在 supply_plan.md。
4. 失败轮不进账（同 CLI）：LLM/工具出错的轮次不写进会话历史，
   避免半截对话污染下一轮的上下文。

运行：python web.py → http://127.0.0.1:8000（.env 需 WEB_API_TOKEN）
"""
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from agent import build_agent, run_turn

BASE_DIR = config.BASE_DIR
SESSIONS_FILE = config.SESSIONS_FILE
HISTORY_ROUNDS = 2  # 金鱼记忆：喂最近 2 轮（见模块注释第 2 条）

config.require_secret("WEB_API_TOKEN")  # L4：fail fast 集中进 config
WEB_API_TOKEN = os.environ["WEB_API_TOKEN"]

# ---------- 结构化日志（L4）：JSON 行格式进 logs/web.log ----------
# 每行一个 JSON 对象（ts/event/session_id/...）——grep 一个 session_id
# 就能把一次对话的全链串起来：chat_start → tool_call → tool_result → chat_end。
# 这是给机器读的账（对账/排查），CLI 的 print trace 是给人看的调试口，两不耽误。

class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # record.msg 是 dict 就直接当载荷（单层 JSON）；是字符串就套个 msg 字段
        core = record.msg if isinstance(record.msg, dict) else {"msg": record.getMessage()}
        return json.dumps({"ts": datetime.now().isoformat(timespec="milliseconds"),
                           "level": record.levelname, **core},
                          ensure_ascii=False)


def _setup_logging() -> logging.Logger:
    config.LOG_DIR.mkdir(exist_ok=True)  # 日志目录自动建，一键起的底气
    logger = logging.getLogger("web")
    handler = logging.FileHandler(config.LOG_DIR / "web.log", encoding="utf-8")
    handler.setFormatter(_JsonLineFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = _setup_logging()


def _log(event: str, session_id: str, **fields) -> None:
    logger.info({"event": event, "session_id": session_id, **fields})


# ---------- sessions.json 账本（读改存纪律，同业务账本） ----------

def _load_sessions() -> list:
    if not SESSIONS_FILE.exists():
        return []
    return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))


def _save_sessions(sessions: list) -> None:
    SESSIONS_FILE.write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _find(sessions: list, sid: str):
    return next((s for s in sessions if s["id"] == sid), None)


def _feed_history(messages: list) -> list:
    """从会话消息里取最近 2 轮纯文本喂 agent（金鱼记忆的取数口）"""
    pairs = []
    for m in messages:
        if m["role"] in ("user", "assistant") and not m.get("error"):
            pairs.append({"role": m["role"], "content": m["content"]})
    return pairs[-HISTORY_ROUNDS * 2:]


# ---------- agent 生命周期：全局一个实例，FastAPI lifespan 管生管死 ----------

_state: dict = {}
_chat_lock = asyncio.Lock()  # 串行化对话轮（见模块注释第 3 条）


@asynccontextmanager
async def lifespan(app: FastAPI):
    role = config.APP_ROLE
    async with build_agent(role) as agent:  # 公共件：CLI 同款后厨
        _state["agent"] = agent
        _state["role"] = role
        yield
    _state.clear()


app = FastAPI(title="供小链 Web", lifespan=lifespan)


async def require_token(authorization: str = Header(None)):
    """外网门的钥匙检查：所有 /api/* 都要过这道（浏览器存 localStorage）"""
    if authorization != f"Bearer {WEB_API_TOKEN}":
        raise HTTPException(status_code=401, detail="无效或缺失的 API Token")


class ChatIn(BaseModel):
    session_id: str
    message: str


@app.post("/api/sessions", dependencies=[Depends(require_token)])
def create_session():
    sid = uuid.uuid4().hex[:12]
    session = {"id": sid, "title": "新会话", "created_at": datetime.now().isoformat(timespec="seconds"), "messages": []}
    sessions = _load_sessions()      # 读一份
    sessions.append(session)         # 改这份
    _save_sessions(sessions)         # 存这份
    return session


@app.get("/api/sessions", dependencies=[Depends(require_token)])
def list_sessions():
    return [{"id": s["id"], "title": s["title"], "created_at": s["created_at"],
             "turns": len([m for m in s["messages"] if m["role"] == "user"])}
            for s in _load_sessions()]


@app.get("/api/sessions/{sid}/messages", dependencies=[Depends(require_token)])
def get_messages(sid: str):
    session = _find(_load_sessions(), sid)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session["messages"]


@app.post("/api/chat", dependencies=[Depends(require_token)])
async def chat(body: ChatIn):
    async with _chat_lock:  # 一轮没跑完下一轮排队——学习版并发答案
        sessions = _load_sessions()
        session = _find(sessions, body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在，请先新建")

        question = body.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="消息不能为空")

        now = datetime.now().isoformat(timespec="seconds")
        t0 = time.perf_counter()
        _log("chat_start", body.session_id, question=question[:80])
        tool_t0 = {}  # 工具计时：call→result 一对，量出每把刀磨了多久

        def _web_event(kind: str, payload) -> None:
            # L3 埋的 on_event 回调在 L4 长成日志管线——工具一动手就记账
            if kind == "call":
                tool_t0["t"] = time.perf_counter()
                _log("tool_call", body.session_id, tool=payload["tool"],
                     args=payload["args"])
            else:
                _log("tool_result", body.session_id, result=str(payload)[:200],
                     duration_ms=round((time.perf_counter() - tool_t0.get("t", t0)) * 1000))

        try:
            answer, _new_msgs, trace = await run_turn(
                _state["agent"], _feed_history(session["messages"]), question,
                on_event=_web_event)
        except Exception as e:
            # 失败轮不进账（见模块注释第 4 条）：错误原样告诉前端，会话不动
            _log("chat_error", body.session_id, error=f"{type(e).__name__}: {e}",
                 duration_ms=round((time.perf_counter() - t0) * 1000))
            return {"answer": f"[出错] {type(e).__name__}: {e}", "trace": [], "error": True}

        if not answer:
            return {"answer": "[出错] 本轮没有产出答案", "trace": trace, "error": True}

        # 成功轮进账：user + assistant（含 trace，回放时折叠展示）
        session["messages"].append({"role": "user", "content": question, "ts": now})
        session["messages"].append({"role": "assistant", "content": answer,
                                    "ts": now, "trace": trace})
        if session["title"] == "新会话":
            session["title"] = question[:20]  # 首问当会话标题
        _save_sessions(sessions)  # 改完这份才存
        _log("chat_end", body.session_id, answer_head=answer[:80],
             duration_ms=round((time.perf_counter() - t0) * 1000))
        return {"answer": answer, "trace": trace}


# 静态门脸挂最后（不抢 /api 的道）；html=True 让 / 自动回 index.html
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

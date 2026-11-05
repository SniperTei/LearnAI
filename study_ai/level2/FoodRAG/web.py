"""web 前端入口：FastAPI 把 ask_core 搬上网（学 drug_rag 的 FastAPI 分层，砍掉 session）。

【设计笔记】

服务端零状态：对话历史（含 lock）和槽位答案全由前端持有，每轮请求原样回传。
好处——服务端不用记账、重启不丢上下文（上下文本来就在浏览器手里）；
代价——前端得自觉维护 history 只留最近 2 轮（金鱼记忆的活从 CLI 挪到了 JS）。

接口四个：
    POST /api/chat           问答主接口（进 question/history/answers，出 ask_core 结果）
    POST /api/meals          记一笔吃饭账（几号吃了什么）
    GET  /api/meals          查近 N 天吃了啥
    POST /api/answer_cache   存标准答案（前端收藏按钮调）
    GET  /api/answer_cache   看缓存账本（调试用）
前两个账本（meal_log/answer_cache）是 JSON 文件——数据小、单用户，读进内存筛足够，
跟项目"万物皆账本"一个风格；哪天要升级 SQLite，只换下面两个 _read/_write 小函数。

静态页挂在根路径（static/index.html），前后端一个端口，不用管跨域。
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ask import ask_core, load_answer_cache, parse_date, save_answer_cache

app = FastAPI(title="金燕美食助手")

DATA_DIR = Path(__file__).parent / "data"
MEAL_LOG_PATH = DATA_DIR / "meal_log.json"  # 吃饭账本：[{date, items, recorded_at}]


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class ChatIn(BaseModel):
    """前端每轮要回传的两本账 + 本轮问题。"""
    question: str
    # 最近 2 轮（问, 答, 锁定的名字）——喂生成层 + 历史带锁，和 CLI 同款
    history: list[dict] = Field(default_factory=list)
    # 槽位累积答案（dine/party/count/area）——反问填槽用，出结果后前端清空
    answers: dict = Field(default_factory=dict)


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    """一轮问答 = 一次 ask_core。缺槽位返回 input，前端拿 prompt 去问用户；
    出答案/没搜到由前端更新历史——服务端只出题不记账。"""
    return ask_core(body.question, history=body.history, answers=body.answers)


# ==== 吃饭账本：几号吃了什么（增删改查四件套）====
# 条目身份证 id = 记账时间戳（微秒级）：删改靠它指认"哪一笔"，
# 不用数组下标（删一笔后面全漂移，指认必错）。老条目没 id 的，读时自动补
# （用它的 recorded_at 当 id——稳定不漂移），补完顺手写回账本。

class MealIn(BaseModel):
    date: str = Field(description="几号吃的，如 11-04 或 2026-11-04（也认'今天/昨天'）")
    items: list[str] = Field(description="吃了啥（菜名/店名列表）")


class MealUpdate(BaseModel):
    """改账只准改这两样；id/记账时间不许动（动了账本就失忆了）。"""
    date: str | None = None
    items: list[str] | None = None


def _load_meals() -> list[dict]:
    """读饭账 + 给老条目补 id。"""
    log: list = _read_json(MEAL_LOG_PATH, [])
    changed = False
    for i, entry in enumerate(log):
        if "id" not in entry:
            entry["id"] = entry.get("recorded_at") or f"row{i}"
            changed = True
    if changed:
        _write_json(MEAL_LOG_PATH, log)
    return log


def _save_meals(log: list[dict]) -> None:
    log.sort(key=lambda e: e["date"])  # 同一天多笔按日期归拢，查时一天的饭一屏看完
    _write_json(MEAL_LOG_PATH, log)


@app.post("/api/meals")
def add_meal(body: MealIn) -> dict:
    date = parse_date(body.date)
    if date is None:
        return {"ok": False, "error": f"认不出日期「{body.date}」，请用 11-04 或 2026-11-04 这种写法"}
    log = _load_meals()
    entry = {"id": datetime.datetime.now().isoformat(timespec="microseconds"),
             "date": date.isoformat(), "items": body.items,
             "recorded_at": datetime.datetime.now().isoformat(timespec="seconds")}
    log.append(entry)
    _save_meals(log)
    return {"ok": True, "id": entry["id"], "date": entry["date"],
            "items": entry["items"], "total": len(log)}


@app.get("/api/meals")
def get_meals(days: int | None = None) -> dict:
    """查吃饭账：?days=7 查近7天，不带参数查全部（认不出的日期条目如实跳过不炸）。"""
    if days is None:
        return {"days": None, "meals": _load_meals()}
    today = datetime.date.today()
    meals = [e for e in _load_meals()
             if (d := parse_date(e["date"])) is not None and (today - d).days < days]
    return {"days": days, "meals": meals}


@app.get("/api/meals/{entry_id}")
def get_meal(entry_id: str) -> dict:
    for entry in _load_meals():
        if entry["id"] == entry_id:
            return {"ok": True, "meal": entry}
    return {"ok": False, "error": f"账本里没有 id={entry_id} 这笔"}


@app.put("/api/meals/{entry_id}")
def update_meal(entry_id: str, body: MealUpdate) -> dict:
    log = _load_meals()  # 读一份、改这份、存这份——中途不能再读（读出来的是没改过的）
    for entry in log:
        if entry["id"] == entry_id:
            if body.date is not None:
                d = parse_date(body.date)
                if d is None:
                    return {"ok": False, "error": f"认不出日期「{body.date}」"}
                entry["date"] = d.isoformat()
            if body.items is not None:
                entry["items"] = body.items
            _save_meals(log)
            return {"ok": True, "meal": entry}
    return {"ok": False, "error": f"账本里没有 id={entry_id} 这笔"}


@app.delete("/api/meals/{entry_id}")
def delete_meal(entry_id: str) -> dict:
    log = _load_meals()
    rest = [e for e in log if e["id"] != entry_id]
    if len(rest) == len(log):
        return {"ok": False, "error": f"账本里没有 id={entry_id} 这笔"}
    _write_json(MEAL_LOG_PATH, rest)
    return {"ok": True, "deleted": entry_id, "total": len(rest)}


# ==== 答案缓存账本：标准答案收藏夹 ====

class CacheIn(BaseModel):
    question: str
    answer: str
    lock: str | None = None            # 收藏时那轮的锁（锅包肉），下轮追问靠它
    sources: list[str] = Field(default_factory=list)  # 来源章一起存，命中时照盖


@app.post("/api/answer_cache")
def add_cache(body: CacheIn) -> dict:
    total = save_answer_cache(body.question, body.answer, body.lock, body.sources)
    return {"ok": True, "total": total}


@app.get("/api/answer_cache")
def list_cache() -> dict:
    """看缓存账本（调试用）：{问题: {answer, lock, sources, saved_at}}。"""
    return load_answer_cache()


# 静态页挂根路径：浏览器开 http://127.0.0.1:8000 直接进聊天界面
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

"""web 前端入口：FastAPI 把 ask_core 搬上网（学 drug_rag 的 FastAPI 分层，砍掉 session）。

【设计笔记】

服务端零状态：对话历史（含 lock）和槽位答案全由前端持有，每轮请求原样回传。
好处——服务端不用记账、重启不丢上下文（上下文本来就在浏览器手里）；
代价——前端得自觉维护 history 只留最近 2 轮（金鱼记忆的活从 CLI 挪到了 JS）。

接口就一个：POST /api/chat，进 {question, history, answers}，出 ask_core 的结果 dict。
静态页挂在根路径（static/index.html），前后端一个端口，不用管跨域。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ask import ask_core

app = FastAPI(title="金燕美食助手")


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


# 静态页挂根路径：浏览器开 http://127.0.0.1:8000 直接进聊天界面
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

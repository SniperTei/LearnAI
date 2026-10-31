"""拼：路由选仓 → 检索证据 → 受控生成（本地 qwen3:4b，一行切换 DashScope）。

【设计笔记】

和 search.py 同款路由：问做法走菜品仓，找店走饭店仓，
路由理由和证据一起打印——答案权哪来的，全程可追溯（trace 思想）。

受控生成（学 drug_rag qa_chain，这条不能砍）：
    证据塞提示词，"只准依据证据回答，没有就说没有"。
    饭店问题和做菜问题共用一套规则，答案末尾标来源。

LLM 可切换：PROVIDER 改成 "dashscope" 即换供应商。
qwen3:4b 是思考模型，可能吐 <think> 段——剥掉再给用户看。
"""

from __future__ import annotations

import argparse
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from router import DB_CONFIG, resolve_db
from search import build_filter, get_store

# ==== 切换开关：ollama / dashscope ====
PROVIDER = "ollama"

SYSTEM_PROMPT = """你是金燕美食助手，负责回答做菜问题和饭店推荐。规则：
1. 只能依据下面提供的证据回答，证据里没有的信息就明说"资料里没有"，不要编造。
2. 做菜问题给出关键步骤和注意点；饭店问题给出店名、人均、地址等关键信息。
3. 涉及多家店/多道菜时逐家列出，不要混在一起。
4. 回答末尾单独一行写"来源：xxx"，只写证据里实际出现的店名或菜名。"""


def build_llm():
    """增强层工厂：一处开关，两家供应商。"""
    if PROVIDER == "ollama":
        return ChatOllama(model="qwen3:4b", temperature=0.1)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="qwen3.7-flash-2026-07-15",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.1,
    )


def _clean(text: str) -> str:
    """剥掉 qwen3 思考模型的 <think> 段。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask(question: str, k: int = 3) -> None:
    # ---- 路由 + 查：复用 search.py 的判单树 ----
    db, reason = resolve_db(question)
    dish_filter = build_filter(question, db, type_filter=None)
    docs = get_store(db).similarity_search_with_relevance_scores(
        question, k=k, filter=dish_filter
    )
    if not docs:
        print("没检索到任何证据，无法回答。")
        return

    label_key = DB_CONFIG[db]["label_key"]
    evidence = "\n\n".join(
        f"【证据{i}】{doc.page_content}" for i, (doc, _) in enumerate(docs, start=1)
    )

    # ---- trace：先看路由和证据，再看答案 ----
    print(f"仓库: {db}（{reason}）")
    print(f"证据: {[doc.metadata[label_key] for doc, _ in docs]}")
    print("=" * 50)

    # ---- 拼：受控生成 ----
    llm = build_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"问题：{question}\n\n证据：\n{evidence}"),
    ]
    answer = _clean(llm.invoke(messages).content)
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金燕美食问答（自动选仓）")
    parser.add_argument("question", help="想问什么")
    parser.add_argument("--k", type=int, default=3, help="取几块证据（默认3）")
    args = parser.parse_args()
    ask(args.question, k=args.k)

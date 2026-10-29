"""拼：检索证据 → 受控生成（本地 qwen3:4b，一行切换 DashScope）。

【设计笔记】

受控生成（学 drug_rag qa_chain 的核心，其他都能砍，这条不能砍）：
    证据塞进提示词，规定"只准依据证据回答，没有就说没有"。
    验收标准：问菜单上没有的菜，必须老实说没有，不许编。

LLM 可切换（增强层可替换，学 drug_rag 的降级哲学）：
    想换 DashScope，把 PROVIDER 改成 "dashscope" 即可，其余代码不动。
    键匙从环境变量 DASHSCOPE_API_KEY 读（学 drug_rag：密钥不进代码）。

qwen3:4b 是思考模型，可能吐 <think> 段 —— 剥掉再给用户看。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from search import CHROMA_DIR, COLLECTION, build_filter, lock_dish

# ==== 切换开关：ollama / dashscope ====
PROVIDER = "ollama"

SYSTEM_PROMPT = """你是金燕餐厅的菜单助手。规则：
1. 只能依据下面提供的菜单证据回答，证据里没有的信息就明说"菜单里没有"，不要编造。
2. 回答简洁实用，做菜问题给出关键步骤和注意点。
3. 回答末尾单独一行写"来源：菜名"，列出用到的菜。"""


def build_llm():
    """增强层工厂：一处开关，两家供应商。"""
    if PROVIDER == "ollama":
        return ChatOllama(model="qwen3:4b", temperature=0.1)
    # DashScope（OpenAI 兼容模式，延迟导入：用 ollama 时不强制装 openai 包）
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
    # ---- 查：复用 search.py 的锁菜+过滤 ----
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=OllamaEmbeddings(model="bge-m3"),
    )
    dish_filter = build_filter(question, dish_type=None)
    docs = store.similarity_search_with_relevance_scores(
        question, k=k, filter=dish_filter
    )
    if not docs:
        print("没检索到任何菜单证据，无法回答。")
        return

    evidence = "\n\n".join(
        f"【证据{i}】{doc.page_content}" for i, (doc, _) in enumerate(docs, start=1)
    )

    # ---- trace：先给用户看检索录像，再看答案 ----
    locked = lock_dish(question)
    print(f"锁菜: {locked if locked else '未锁定'}")
    print(f"证据: {[doc.metadata['菜名'] for doc, _ in docs]}")
    print("=" * 50)

    # ---- 拼：受控生成 ----
    llm = build_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"问题：{question}\n\n菜单证据：\n{evidence}"),
    ]
    answer = _clean(llm.invoke(messages).content)
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金燕菜单问答")
    parser.add_argument("question", help="想问什么，如：锅包肉怎么做")
    parser.add_argument("--k", type=int, default=3, help="取几块证据（默认3）")
    args = parser.parse_args()
    ask(args.question, k=args.k)

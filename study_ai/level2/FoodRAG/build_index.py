"""量化 + 存：账本 61 块 → Chroma 仓库（本地 Ollama bge-m3 嵌入）。

【设计笔记】

量化没有独立的步骤代码——学 drug_rag 的外包套路：
    构造 Chroma 时塞进 embedding_function，add_texts 时自动量化。

先删后存 = 幂等（学 drug_rag 的 vector_index）：
    重复跑这个脚本，仓库里永远只有 61 块，不会翻倍。

id 用菜名：
    一道菜一块，菜名天然唯一（学 drug_rag 的 chunk_id 身份证思路）。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_DIR = Path(__file__).parent / "data"
IN_JSONL = DATA_DIR / "menu_chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "menu_chunks"

# 本地嵌入：免费、无限量、任何终端都能跑。
# 将来要换 DashScope，只改这一处（换成对应的 Embeddings 类）。
EMBEDDINGS = OllamaEmbeddings(model="bge-m3")


def load_ledger() -> list[dict]:
    """读账本：一行一块，还原 page_content + metadata。"""
    rows = []
    for line in IN_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_index() -> Chroma:
    rows = load_ledger()
    texts = [row["page_content"] for row in rows]
    metadatas = [row["metadata"] for row in rows]
    ids = [f"menu:{row['metadata']['菜名']}" for row in rows]

    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=EMBEDDINGS,
    )
    # 先删后存：幂等
    try:
        store.delete(ids=ids)
    except Exception:
        pass  # 首次运行仓库是空的，删不到不报错
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return store


if __name__ == "__main__":
    store = build_index()

    # ---- 验收 ----
    count = store._collection.count()
    print(f"仓库块数: {count}")

    # 冒烟测试：搜一个"清淡的汤"，看向量搜索灵不灵（顺便预演题3）
    print('\n冒烟测试: similarity_search("有什么清淡的汤", k=3)')
    for doc in store.similarity_search("有什么清淡的汤", k=3):
        print(f"  - {doc.metadata['菜名']}（{doc.metadata['类型']} / {doc.metadata['口味']}）")

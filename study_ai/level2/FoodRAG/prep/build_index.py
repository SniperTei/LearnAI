"""量化 + 存：两本账本 → 两个 Chroma 仓库（本地 Ollama bge-m3 嵌入）。

【设计笔记】

量化没有独立的步骤代码——学 drug_rag 的外包套路：
    构造 Chroma 时塞进 embedding_function，add_texts 时自动量化。

先删后存 = 幂等（学 drug_rag 的 vector_index）：
    重复跑这个脚本，每个仓库块数不变，不会翻倍。

双仓库分治（学 drug_rag 的 drug_chunks / drug_summaries 分仓思路）：
    menu_chunks      菜品仓：61 道菜
    restaurant_chunks 饭店仓：6 家店（一分店一块）
    两仓独立过滤、独立检索——将来"找店"和"问做法"各查各的仓，互不污染。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"  # prep/ 的上一层才是项目根
CHROMA_DIR = DATA_DIR / "chroma"

# 本地嵌入：免费、无限量、任何终端都能跑。
# 将来要换 DashScope，只改这一处（换成对应的 Embeddings 类）。
EMBEDDINGS = OllamaEmbeddings(model="bge-m3")

# 入账本 → 出仓库：一张登记表管两条备菜线
# 块 id 用身份证（c001/s001）：菜改名/店换分店名，块的 id 不变——
# 旧 id 的引用（收藏的来源、将来 CRUD）不因改名断线。
BUILD_PLAN = [
    {"jsonl": DATA_DIR / "menu_chunks.jsonl", "collection": "menu_chunks",
     "id_key": "id", "smoke_query": "有什么清淡的汤"},
    {"jsonl": DATA_DIR / "restaurant_chunks.jsonl", "collection": "restaurant_chunks",
     "id_key": "id", "smoke_query": "罗湖万象城有什么便宜的工作餐"},
]


def load_ledger(jsonl_path: Path) -> list[dict]:
    """读账本：一行一块，还原 page_content + metadata。"""
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_one(jsonl_path: Path, collection: str, id_key: str) -> Chroma:
    rows = load_ledger(jsonl_path)
    texts = [row["page_content"] for row in rows]
    metadatas = [row["metadata"] for row in rows]
    # id 来源：两仓都用账本身份证（backfill_ids 发的 c001/s001），前缀仅防串仓
    ids = [f"{collection}:{row['metadata'][id_key]}" for row in rows]

    # 整仓删除重建：mini 规模最稳的幂等。
    # （曾用"先删后存"按 id 删，但 id 规则一改旧块就成了删不掉的僵尸，仓库翻倍成122——
    #  验收块数当场逮住。整仓重建彻底不怕 id 规则变化，代价是每次全量重嵌，61+6块毫秒级无所谓）
    client_store = Chroma(
        collection_name=collection,
        persist_directory=str(CHROMA_DIR),
        embedding_function=EMBEDDINGS,
    )
    try:
        client_store.delete_collection()
    except Exception:
        pass  # 首次运行仓库不存在
    store = Chroma(
        collection_name=collection,
        persist_directory=str(CHROMA_DIR),
        embedding_function=EMBEDDINGS,
    )
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return store


if __name__ == "__main__":
    for plan in BUILD_PLAN:
        store = build_one(plan["jsonl"], plan["collection"], plan["id_key"])

        # ---- 验收：块数 + 冒烟测试 ----
        count = store._collection.count()
        print(f"\n仓库 {plan['collection']}: {count} 块")
        print(f'冒烟测试: similarity_search("{plan["smoke_query"]}", k=3)')
        for doc in store.similarity_search(plan["smoke_query"], k=3):
            meta = doc.metadata
            label = meta.get("展示名") or meta.get("菜名")
            print(f"  - {label}（{meta.get('品类') or meta.get('类型')} / 人均{meta.get('人均', '')}）")

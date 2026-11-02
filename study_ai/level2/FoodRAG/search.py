"""查：锁定 + 向量检索 + trace 打印。（"去哪查"的决策在 router.py）

【设计笔记】

锁定 = 迷你版"锁药"，双仓各锁各的名字：
    菜品仓锁菜名（唯一命中才锁），饭店仓锁店名（可多店名 $in；
    同名连锁靠"店名=品牌"天然全召回，两家分店一起列出 = 迷你版多命中处理）。
"""

from __future__ import annotations

import argparse
import json
import time

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from router import CHROMA_DIR, DB_CONFIG, extract_max_price, load_rows, match_names, resolve_db

EMBEDDINGS = OllamaEmbeddings(model="bge-m3")


def build_filter(question: str, db: str, type_filter: str | None, max_price: int | None = None) -> dict | None:
    """过滤卡：片子拼装——名字片/类型片/价格片，单片不包壳、多片包 $and
    （拼装规矩学 drug_rag build_metadata_filter，第3关验证过：单片直接用，多片才包壳）。

    价格片两条来路：--max-price 手动指定优先，没传就从问题里自动提取
    （"人均50以内"→50）。只给饭店仓加：菜品仓 metadata 没有"人均"字段，
    加了必 0 命中。
    """
    cfg = DB_CONFIG[db]
    if max_price is None:
        max_price = extract_max_price(question)

    pieces: list[dict] = []
    hits = match_names(question, [r["metadata"][cfg["name_key"]] for r in load_rows(db)])
    if hits:
        pieces.append({cfg["name_key"]: hits[0] if len(hits) == 1 else {"$in": hits}})
    if type_filter:
        pieces.append({cfg["type_key"]: type_filter})
    if max_price is not None and db == "restaurant":
        pieces.append({"人均": {"$lte": max_price}})

    if not pieces:
        return None
    return pieces[0] if len(pieces) == 1 else {"$and": pieces}


def get_store(db: str) -> Chroma:
    return Chroma(
        collection_name=DB_CONFIG[db]["collection"],
        persist_directory=str(CHROMA_DIR),
        embedding_function=EMBEDDINGS,
    )


def search(question: str, db: str | None = None, type_filter: str | None = None,
           k: int = 3, max_price: int | None = None) -> None:
    # ---- 路由：--db 手动指定优先（调试后门），否则走判单树 ----
    if db is None:
        db, reason = resolve_db(question)
    else:
        reason = "手动指定 --db"
    dish_filter = build_filter(question, db, type_filter, max_price=max_price)

    # ---- trace 打印（监控录像：路由→filter→战果）----
    print(f"问题: {question}")
    print(f"仓库: {db}（{reason}）")
    print(f"filter: {json.dumps(dish_filter, ensure_ascii=False) if dish_filter else '无'}")
    print("-" * 50)

    # ---- 计时：只量检索本身（perf_counter 单调钟，专管测耗时）----
    t0 = time.perf_counter()
    results = get_store(db).similarity_search_with_relevance_scores(
        question, k=k, filter=dish_filter
    )
    elapsed = time.perf_counter() - t0
    if not results:
        print("0 命中！检查 filter 条件。")
        return
    print(f"耗时: {elapsed:.3f}秒（向量检索 {len(results)} 块）")
    label_key = DB_CONFIG[db]["label_key"]
    for doc, score in results:
        meta = doc.metadata
        extra = f" / 人均{meta['人均']}" if "人均" in meta else f" / {meta.get('口味', '')}"
        print(f"[{score:.3f}] {meta[label_key]}（{meta.get('类型') or meta.get('品类')}{extra}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金燕美食检索（自动选仓）")
    parser.add_argument("question", help="想问什么")
    parser.add_argument("--db", choices=["menu", "restaurant"], default=None,
                        help="手动指定仓库（默认自动路由）")
    parser.add_argument("--type", default=None,
                        help="按类型过滤：菜品仓填类型（荤菜/素菜…），饭店仓填品类（烤串/烧腊…）")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-price", type=int, default=None,
                        help="人均上限（如 --max-price 50 → 人均≤50，仅饭店仓生效）")
    args = parser.parse_args()
    search(args.question, db=args.db, type_filter=args.type, k=args.k, max_price=args.max_price)

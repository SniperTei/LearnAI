"""查：锁菜（学 drug_rag 锁药）+ 向量检索 + trace 打印。

【设计笔记】

锁菜 = 药品目录锁药的迷你版：
    马甲 = 菜名本身（drug_rag 有3个马甲：商品名/通用名/文件名，菜单只有一个名字）
    匹配方向 = "菜名 in 问题"（整支菜名必须原样出现在问题里）
    —— 和 drug_rag 同一条铁律：问"锅包"俩字匹配不上"锅包肉"全名

锁菜的目的（学 drug_rag 的教训，这次提前防御）：
    问"锅包肉怎么做"时，纯向量搜索会把糖醋排骨/锅包豆腐这些"酸甜炸物亲戚"
    一起捞回来——不同菜的做法不可混用，等于 drug_rag 的"混药事故"。
    锁菜后 filter={菜名: 锅包肉}，精确到一块。

trace 打印 = 采购监控录像迷你版：拍到"锁到没有/用什么filter/捞回什么"。

已知局限（留给升级）：
    ① 别名锁不到——"孜然牛肉"是孜盐牛肉的备注别名，但马甲表只有菜名，
      问"孜然牛肉怎么做"锁不上（drug_rag 的3马甲就是为这种场景准备的）
    ② 多命中不锁——问"锅包肉和锅包豆腐哪个好"两个名字都命中，直接不锁+提示
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_DIR = Path(__file__).parent / "data"
IN_JSONL = DATA_DIR / "menu_chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "menu_chunks"

EMBEDDINGS = OllamaEmbeddings(model="bge-m3")


def _normalize(text: str) -> str:
    """卸妆间（学 drug_rag metadata_index._normalize）：
    转小写、删空格、统一全半角括号——让"수육（水煮肉）"的两个版本对上眼。"""
    return (
        str(text)
        .lower()
        .replace(" ", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def load_dish_names() -> list[str]:
    """从账本读全部菜名 = 迷你版名牌墙。"""
    names = []
    for line in IN_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.append(json.loads(line)["metadata"]["菜名"])
    return names


def lock_dish(question: str) -> str | None:
    """锁菜：菜名 in 问题。唯一命中才锁，0命中或多命中都不锁。"""
    normalized_question = _normalize(question)
    hits = [
        name for name in load_dish_names()
        if _normalize(name) in normalized_question
    ]
    if len(set(hits)) == 1:
        return hits[0]
    return None


def build_filter(question: str, dish_type: str | None) -> dict | None:
    """过滤卡：锁菜片优先，其次类型片，都没有就裸搜。"""
    locked = lock_dish(question)
    if locked:
        return {"菜名": locked}
    if dish_type:
        return {"类型": dish_type}
    return None


def search(question: str, dish_type: str | None = None, k: int = 3) -> None:
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=EMBEDDINGS,
    )
    dish_filter = build_filter(question, dish_type)

    # ---- trace 打印（监控录像迷你版）----
    locked = lock_dish(question)
    print(f"问题: {question}")
    print(f"锁菜: {locked if locked else '未锁定（无过滤卡，裸搜）'}")
    print(f"filter: {json.dumps(dish_filter, ensure_ascii=False) if dish_filter else '无'}")
    print("-" * 50)

    results = store.similarity_search_with_relevance_scores(
        question, k=k, filter=dish_filter
    )
    if not results:
        print("0 命中！检查 filter 条件是否写死了一块不存在的菜。")
        return
    for doc, score in results:
        meta = doc.metadata
        print(f"[{score:.3f}] {meta['菜名']}（{meta['类型']} / {meta['口味']}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="金燕菜单检索")
    parser.add_argument("question", help="想问什么，如：锅包肉怎么做")
    parser.add_argument("--type", default=None, help="可选：按类型过滤（荤菜/素菜/凉菜/汤...）")
    parser.add_argument("--k", type=int, default=3, help="取回几块（默认3）")
    args = parser.parse_args()
    search(args.question, dish_type=args.type, k=args.k)

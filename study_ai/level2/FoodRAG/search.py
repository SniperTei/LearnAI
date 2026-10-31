"""查：选仓路由（迷你版 intent_router）+ 锁定 + 向量检索 + trace 打印。

【设计笔记】

选仓判单树（学 drug_rag 判单树：按优先级排队，先命中先走）：

    ① 店名命中问题 → 饭店仓     （"大富烧鸡怎么样"——名字是最强信号）
    ② 菜名命中问题 → 菜品仓     （"锅包肉怎么做"）
    ③ 场景词：外食系 → 饭店仓   （外食/出去吃/在哪吃/哪家店/下馆子…）
       场景词：家饭系 → 菜品仓   （家饭/在家吃/自己做…）
    ④ 商圈词命中   → 饭店仓     （"罗湖万象城有什么工作餐"——商圈值从账本现拉，
                                   账本加新商圈，词表自动跟着长）
    ⑤ 兜底 → 菜品仓             （drug_rag 教训：兜底必有，表里没写的问法落这里）

和 drug_rag intent_router 同一个命门：词表没写的说法会漏网
（"想吃烤肉"没店名没场景词→兜底走错菜品仓）。mini 版先接受，
将来升级路 = drug_rag 同款：LLM 二次修正，或扩词表。

锁定 = 迷你版"锁药"，双仓各锁各的名字：
    菜品仓锁菜名（唯一命中才锁），饭店仓锁店名（可多店名 $in；
    同名连锁靠"店名=品牌"天然全召回，两家分店一起列出 = 迷你版多命中处理）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"

# 场景词表（用户原创词表 + 少量补充）
SCENE_RESTAURANT_WORDS = ["外食", "出去吃", "在哪吃", "哪里吃", "哪家店", "下馆子", "饭店"]
SCENE_MENU_WORDS = ["家饭", "在家吃", "自己做", "家常"]

EMBEDDINGS = OllamaEmbeddings(model="bge-m3")

DB_CONFIG = {
    "menu": {"collection": "menu_chunks", "jsonl": DATA_DIR / "menu_chunks.jsonl",
             "name_key": "菜名", "label_key": "菜名", "type_key": "类型"},
    "restaurant": {"collection": "restaurant_chunks", "jsonl": DATA_DIR / "restaurant_chunks.jsonl",
                   "name_key": "店名", "label_key": "展示名", "type_key": "品类"},
}


def _normalize(text: str) -> str:
    """卸妆：转小写、删空格、统一全半角括号（学 drug_rag metadata_index）。"""
    return str(text).lower().replace(" ", "").replace("（", "(").replace("）", ")")


def _load_rows(db: str) -> list[dict]:
    rows = []
    for line in DB_CONFIG[db]["jsonl"].read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _match_names(question: str, names: list[str]) -> list[str]:
    """名字 in 问题，去重保序（匹配方向学 drug_rag：全名必须原样出现）。"""
    q = _normalize(question)
    seen, hits = set(), []
    for name in names:
        if name and _normalize(name) in q and name not in seen:
            seen.add(name)
            hits.append(name)
    return hits


def resolve_db(question: str) -> tuple[str, str]:
    """选仓判单树：返回 (仓库名, 路由理由)——理由进 trace，路由不再黑盒。"""
    shop_hits = _match_names(question, [r["metadata"]["店名"] for r in _load_rows("restaurant")])
    if shop_hits:
        return "restaurant", f"命中店名: {shop_hits}"
    dish_hits = _match_names(question, [r["metadata"]["菜名"] for r in _load_rows("menu")])
    if dish_hits:
        return "menu", f"命中菜名: {dish_hits}"
    q = _normalize(question)
    if any(w in q for w in SCENE_RESTAURANT_WORDS):
        return "restaurant", "场景词: 外食系"
    if any(w in q for w in SCENE_MENU_WORDS):
        return "menu", "场景词: 家饭系"
    area_hits = _match_names(question, [r["metadata"]["商圈"] for r in _load_rows("restaurant") if r["metadata"].get("商圈")])
    if area_hits:
        return "restaurant", f"命中商圈: {area_hits}"
    return "menu", "兜底: 默认菜品仓"


def build_filter(question: str, db: str, type_filter: str | None) -> dict | None:
    """过滤卡：名字片（可多值 $in）优先，其次类型/品类片。"""
    cfg = DB_CONFIG[db]
    hits = _match_names(question, [r["metadata"][cfg["name_key"]] for r in _load_rows(db)])
    if hits:
        return {cfg["name_key"]: hits[0] if len(hits) == 1 else {"$in": hits}}
    if type_filter:
        return {cfg["type_key"]: type_filter}
    return None


def get_store(db: str) -> Chroma:
    return Chroma(
        collection_name=DB_CONFIG[db]["collection"],
        persist_directory=str(CHROMA_DIR),
        embedding_function=EMBEDDINGS,
    )


def search(question: str, db: str | None = None, type_filter: str | None = None, k: int = 3) -> None:
    # ---- 路由：--db 手动指定优先（调试后门），否则走判单树 ----
    if db is None:
        db, reason = resolve_db(question)
    else:
        reason = "手动指定 --db"
    dish_filter = build_filter(question, db, type_filter)

    # ---- trace 打印（监控录像：路由→锁定→filter→战果）----
    print(f"问题: {question}")
    print(f"仓库: {db}（{reason}）")
    print(f"filter: {json.dumps(dish_filter, ensure_ascii=False) if dish_filter else '无'}")
    print("-" * 50)

    results = get_store(db).similarity_search_with_relevance_scores(
        question, k=k, filter=dish_filter
    )
    if not results:
        print("0 命中！检查 filter 条件。")
        return
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
    args = parser.parse_args()
    search(args.question, db=args.db, type_filter=args.type, k=args.k)

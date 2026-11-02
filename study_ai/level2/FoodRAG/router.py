"""路由（迷你版 intent_router，学 drug_rag 独立文件的架构）：

    "去哪查"的决策归这个文件，"怎么查"归 search.py。

【设计笔记】

判单树（按优先级排队，先命中先走）：

    ① 店名命中问题 → 饭店仓     （"大富烧鸡怎么样"——名字是最强信号）
    ② 菜名命中问题 → 菜品仓     （"锅包肉怎么做"）
    ③ 场景词：外食系 → 饭店仓   （外食/出去吃/在哪吃/哪家店/下馆子…）
       场景词：家饭系 → 菜品仓   （家饭/在家吃/自己做…）
    ④ 商圈词命中   → 饭店仓     （商圈值从账本现拉，账本加新商圈词表自动长）
    ⑤ 兜底 → 菜品仓             （drug_rag 教训：兜底必有，词表没写的落这里）

和 drug_rag intent_router 同一个命门：词表没写的说法漏网
（"想吃烤肉"没店名没场景词→兜底走错仓）。升级路 = 扩词表 或 LLM 二次修正。

独立成文件的理由：search 和 ask 两个入口都用它（共享信号），
且词表会长、将来可能挂 LLM 修正（成长信号）——两条都占了。
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"

# 场景词表（用户原创词表 + 少量补充；"加班/工作餐"=10-31实战补词：
# "明天加班中午吃什么"漏到兜底走错仓，用户点名加班该找饭店工作餐）
SCENE_RESTAURANT_WORDS = ["外食", "出去吃", "在哪吃", "哪里吃", "哪家店", "下馆子", "饭店",
                          "加班", "工作餐", "工作日午餐"]
SCENE_MENU_WORDS = ["家饭", "在家吃", "自己做", "家常"]

# 两仓地图：谁叫什么集合、账本在哪、名字/标签字段叫什么
DB_CONFIG = {
    "menu": {"collection": "menu_chunks", "jsonl": DATA_DIR / "menu_chunks.jsonl",
             "name_key": "菜名", "label_key": "菜名", "type_key": "类型"},
    "restaurant": {"collection": "restaurant_chunks", "jsonl": DATA_DIR / "restaurant_chunks.jsonl",
                   "name_key": "店名", "label_key": "展示名", "type_key": "品类"},
}


def normalize(text: str) -> str:
    """卸妆：转小写、删空格、统一全半角括号（学 drug_rag metadata_index）。"""
    return str(text).lower().replace(" ", "").replace("（", "(").replace("）", ")")


def load_rows(db: str) -> list[dict]:
    """读账本：一行一块。"""
    rows = []
    for line in DB_CONFIG[db]["jsonl"].read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def match_names(question: str, names: list[str]) -> list[str]:
    """名字 in 问题，去重保序（匹配方向学 drug_rag：全名必须原样出现）。"""
    q = normalize(question)
    seen, hits = set(), []
    for name in names:
        if name and normalize(name) in q and name not in seen:
            seen.add(name)
            hits.append(name)
    return hits


def resolve_db(question: str) -> tuple[str, str]:
    """选仓判单树：返回 (仓库名, 路由理由)——理由进 trace，路由不黑盒。"""
    shop_hits = match_names(question, [r["metadata"]["店名"] for r in load_rows("restaurant")])
    if shop_hits:
        return "restaurant", f"命中店名: {shop_hits}"
    dish_hits = match_names(question, [r["metadata"]["菜名"] for r in load_rows("menu")])
    if dish_hits:
        return "menu", f"命中菜名: {dish_hits}"
    q = normalize(question)
    if any(w in q for w in SCENE_RESTAURANT_WORDS):
        return "restaurant", "场景词: 外食系"
    if any(w in q for w in SCENE_MENU_WORDS):
        return "menu", "场景词: 家饭系"
    area_hits = match_names(
        question,
        [r["metadata"]["商圈"] for r in load_rows("restaurant") if r["metadata"].get("商圈")],
    )
    if area_hits:
        return "restaurant", f"命中商圈: {area_hits}"
    return "menu", "兜底: 默认菜品仓"

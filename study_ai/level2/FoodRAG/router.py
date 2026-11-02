"""路由（迷你版 intent_router，学 drug_rag 独立文件的架构）：

    "去哪查"的决策归这个文件，"怎么查"归 search.py。

【设计笔记】

判单树（按优先级排队，先命中先走）：

    ① 店名命中问题 → 饭店仓     （"大富烧鸡怎么样"——名字是最强信号）
    ② 菜名命中问题 → 菜品仓     （"锅包肉怎么做"）
    ③ 场景词：外食系 → 饭店仓   （外食/出去吃/在哪吃/哪家店/下馆子…）
       场景词：外卖系 → 饭店仓   （外卖/送餐/配送…——点外卖=从饭店订餐）
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
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"

# 场景词表（用户原创词表 + 少量补充；"加班/工作餐"=10-31实战补词：
# "明天加班中午吃什么"漏到兜底走错仓，用户点名加班该找饭店工作餐）
SCENE_RESTAURANT_WORDS = ["外食", "出去吃", "在哪吃", "哪里吃", "去哪吃", "哪家店", "下馆子",
                          "饭店", "加班", "工作餐", "工作日午餐", "人均"]  # "人均"是饭店词汇，出现即外食信号
SCENE_MENU_WORDS = ["家饭", "在家吃", "自己做", "家常"]
# 外卖系：点外卖=从饭店订餐，目的地仍是饭店仓（子串匹配，"外卖"已覆盖"点外卖"）。
# 懒人系词汇（不想做饭/不想动弹…）= 想点外卖的委婉说法。
# ⚠️ 判单树里外卖系必须排在家饭系【前面】："不想自己做"含家饭词"自己做"，
#    家饭系先查会把"不想自己做"截胡走错方向——词序就是优先级。
# 注意：仓里有没写"能不能送"是另一回事——那是随地址漂移的运行时状态，不进账本（用户拍板）。
SCENE_TAKEOUT_WORDS = ["外卖", "送餐", "配送", "不想做饭", "不想自己做", "不想在家做", "不想动弹"]

# 两仓地图：谁叫什么集合、账本在哪、名字/标签字段叫什么
DB_CONFIG = {
    "menu": {"collection": "menu_chunks", "jsonl": DATA_DIR / "menu_chunks.jsonl",
             "name_key": "菜名", "label_key": "菜名", "type_key": "类型", "alias_key": "别名"},
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


def match_db_names(question: str, db: str) -> list[str]:
    """带别名的名字匹配：正式名/别名都算命中，命中后归一成正式名。

    "孜然牛肉怎么做" → 命中别名"孜然牛肉" → 归一成正式菜名"孜盐牛肉"。
    别名从账本 metadata 的别名字段现拉（顿号/逗号分隔），账本加新别名自动生效。
    """
    cfg = DB_CONFIG[db]
    alias_map: dict[str, str] = {}
    for r in load_rows(db):
        canonical = r["metadata"][cfg["name_key"]]
        alias_map.setdefault(canonical, canonical)
        raw = str(r["metadata"].get(cfg.get("alias_key", ""), "") or "")
        for alias in re.split(r"[、，,]", raw):
            if alias := alias.strip():
                alias_map.setdefault(alias, canonical)
    hits = match_names(question, list(alias_map))
    return list(dict.fromkeys(alias_map[h] for h in hits))  # 多别名同主→归一后去重


# 价格说法模式表：命中任意一个，捕获组里的数字就是价格上限。
# 只认"以内/以下/不超过"这类明确上界说法——"人均50左右"是模糊词，不硬猜（宁漏勿错）。
PRICE_PATTERNS = [
    r"人均(\d+)(?:元|块)?(?:以内|之内|以下)",   # 人均50以内 / 人均50元以内
    r"(\d+)(?:元|块)(?:以内|之内|以下)",         # 50元以内 / 50块以内
    r"不超过(\d+)(?:元|块)?",                   # 不超过50 / 不超过50元
]


def extract_max_price(question: str) -> int | None:
    """从问题里挖价格上限（迷你版 drug_rag resolve_query_constraints 的数值片）。

    人均在 metadata 里存的是 int——备菜线当初就为今天这步铺的路，
    现在才能用 Chroma 的 $lte 数值比较（字符串只能等值匹配）。
    """
    q = normalize(question)
    for pattern in PRICE_PATTERNS:
        m = re.search(pattern, q)
        if m:
            return int(m.group(1))
    return None


# 难度词表：想要"做起来简单"→过滤难度=简单；点名要硬菜→费工夫。
# 和场景词表同一个命门：没写的说法不触发（"不费劲"就没写，漏网再补）。
DIFFICULTY_WORDS = {
    "简单": ["简单", "快手", "省事", "省时间", "方便", "容易"],
    "费工夫": ["费工夫", "费时间", "复杂", "硬菜"],
}


def extract_difficulty(question: str) -> str | None:
    """从问题里挖难度偏好（"工作日晚餐来个简单的" → 简单）。"""
    q = normalize(question)
    for level, words in DIFFICULTY_WORDS.items():
        if any(w in q for w in words):
            return level
    return None


def match_areas(text: str) -> list[str]:
    """商圈匹配：商圈值 in 文本。路由第④分支和外卖地址槽共用——
    问题里带商圈、用户反问答的地址，都用它认（"坂田"→坂田）。"""
    areas = [r["metadata"]["商圈"] for r in load_rows("restaurant") if r["metadata"].get("商圈")]
    return match_names(text, areas)


def resolve_db(question: str) -> tuple[str, str]:
    """选仓判单树：返回 (仓库名, 路由理由)——理由进 trace，路由不黑盒。"""
    shop_hits = match_names(question, [r["metadata"]["店名"] for r in load_rows("restaurant")])
    if shop_hits:
        return "restaurant", f"命中店名: {shop_hits}"
    dish_hits = match_db_names(question, "menu")
    if dish_hits:
        return "menu", f"命中菜名: {dish_hits}"
    q = normalize(question)
    if any(w in q for w in SCENE_RESTAURANT_WORDS):
        return "restaurant", "场景词: 外食系"
    if any(w in q for w in SCENE_TAKEOUT_WORDS):
        return "restaurant", "场景词: 外卖系"
    if any(w in q for w in SCENE_MENU_WORDS):
        return "menu", "场景词: 家饭系"
    area_hits = match_areas(question)
    if area_hits:
        return "restaurant", f"命中商圈: {area_hits}"
    return "menu", "兜底: 默认菜品仓"

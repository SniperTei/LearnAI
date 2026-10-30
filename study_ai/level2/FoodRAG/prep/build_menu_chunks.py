"""读 + 切：61 道菜 → 61 块带标签的 chunk，落盘 JSONL 账本。

【设计笔记】字段分家的判断方法 = 下游倒推（用户将来会问什么）：

进 metadata（标签，给将来精确过滤用）：
    菜名 / 类型 / 口味 / 厨师 / 图片
    —— 短、可枚举，适合当过滤条件（"来个清淡的汤" → 类型=汤）
    —— 菜名进标签 = 学 drug_rag 的"锁药"：将来"锅包肉怎么做"可以精确锁菜

进 page_content（正文，给向量搜索用）：
    做法 / 食材 / 调料 / 注意点 / 描述 / 备注
    —— 长文本，靠语义命中（"有什么酸甜口的下饭菜"）

暗桩技巧（学 drug_rag 摘要卡前缀）：
    正文开头把 菜名+类型+口味 再缝一遍。
    因为"清淡"这种词如果只躺在 metadata 里，向量搜索根本搜不到——
    metadata 是给 filter 看的，向量只认正文。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

DATA_DIR = Path(__file__).resolve().parents[1] / "data"  # prep/ 的上一层才是项目根
MENU_JSON = DATA_DIR / "jinyan_menu.json"
OUT_JSONL = DATA_DIR / "menu_chunks.jsonl"


def build_chunks() -> list[Document]:
    """读入 61 道菜，一道菜 = 一块 chunk（永不跨菜，学 drug_rag 永不跨章节）。"""
    dishes = json.loads(MENU_JSON.read_text(encoding="utf-8"))

    docs: list[Document] = []
    for dish in dishes:
        name = str(dish.get("菜名", "")).strip()
        dish_type = str(dish.get("菜的类型", "")).strip()
        taste = str(dish.get("口味", "")).strip()

        # 正文 = 暗桩前缀 + 各字段正文
        body_lines = [
            f"{name} {dish_type} {taste}",  # 暗桩：让"清淡/荤菜"这类词也进正文
            f"食材：{dish.get('食材', '')}",
            f"调料：{dish.get('调料', '')}",
            f"做法：{dish.get('菜的做法', '')}",
            f"注意点：{dish.get('菜的注意点', '')}",
            f"描述：{dish.get('菜的描述', '')}",
        ]
        remark = str(dish.get("备注", "")).strip()
        if remark:
            # 备注 = 别名/俗名（"孜盐牛肉其实是孜然牛肉"），搜口误也能命中
            body_lines.append(f"备注：{remark}")

        docs.append(
            Document(
                page_content="\n".join(body_lines),
                metadata={
                    "菜名": name,
                    "类型": dish_type,
                    "口味": taste,
                    "厨师": str(dish.get("厨师", "")).strip(),
                    "图片": str(dish.get("菜的图片路径", "")).strip(),
                },
            )
        )
    return docs


def save_jsonl(docs: list[Document], out_path: Path) -> None:
    """一行 = 一块的账本（学 drug_rag 的 data/processed/*.jsonl）。"""
    with out_path.open("w", encoding="utf-8") as f:  # encoding 必须写！
        for doc in docs:
            row = {"page_content": doc.page_content, "metadata": doc.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    all_docs = build_chunks()
    save_jsonl(all_docs, OUT_JSONL)

    # ---- 验收 ----
    print(f"总块数: {len(all_docs)}")
    for doc in all_docs:
        if doc.metadata["菜名"] == "孜盐牛肉":
            print("\n--- 抽查：孜盐牛肉这一块 ---")
            print("metadata:", json.dumps(doc.metadata, ensure_ascii=False))
            print("正文:")
            print(doc.page_content)
            break

"""读 + 切：restaurant.json 6 家店 → 6 块 chunk，落盘 restaurant_chunks.jsonl。

【设计笔记】（对照 build_menu_chunks.py，套路同构）

一分店一块（不是"一品牌一块"）：
    学 drug_rag"聚合键是 doc_id 不是药名"——同名分店是不同的实体。
    大富烧鸡两行 = 两块，各有各的 id 和地址。

字段分家（下游倒推）：
    metadata：店名/分店/品类/人均/商圈/推荐人
      —— 人均保持 int（converter 已转），将来 Chroma 支持 filter 人均 ≤ 100 数值比较
    正文：招牌菜/特点评价/避雷点/营业时间/地址（长文本，靠语义命中）

暗桩前缀：
    正文首行缝「店名 分店 品类 商圈 适合场景」——
    "罗湖万象城有什么工作餐"这类问法，向量直接撞前缀就中，不必读到正文深处。

id 带分店：
    restaurant:大富烧鸡:锦绣店 / restaurant:大富烧鸡:坂田店 ——连锁同名靠分店区分，
    同款先删后存幂等。
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

DATA_DIR = Path(__file__).resolve().parents[1] / "data"  # prep/ 的上一层才是项目根
IN_JSON = DATA_DIR / "restaurant.json"
OUT_JSONL = DATA_DIR / "restaurant_chunks.jsonl"


def build_chunks() -> list[Document]:
    shops = json.loads(IN_JSON.read_text(encoding="utf-8"))

    docs: list[Document] = []
    for shop in shops:
        name = shop["店名"]
        branch = shop["分店"]
        display_name = f"{name}（{branch}）" if branch else name  # 展示用全名

        body_lines = [
            f"{display_name} {shop['品类']} {shop['商圈']} {shop['适合场景']}",  # 暗桩前缀
            f"人均：{shop['人均']}元",  # 人均必须进正文：metadata 只给过滤，LLM 只看得见正文
            f"招牌菜：{shop['招牌菜']}" if shop["招牌菜"] else None,
            f"特点评价：{shop['特点评价']}" if shop["特点评价"] else None,
            f"避雷点：{shop['避雷点']}" if shop["避雷点"] else None,
            f"营业时间：{shop['营业时间']}" if shop["营业时间"] else None,
            f"地址：{shop['地址']}" if shop["地址"] else None,
        ]
        body = "\n".join(line for line in body_lines if line)

        doc_id = f"restaurant:{name}:{branch}" if branch else f"restaurant:{name}"
        docs.append(
            Document(
                page_content=body,
                metadata={
                    "店名": name,
                    "分店": branch,
                    "展示名": display_name,
                    "品类": shop["品类"],
                    "人均": shop["人均"],  # int，Chroma 数值过滤用
                    "商圈": shop["商圈"],
                    "推荐人": shop["推荐人"],
                    "chunk_id": doc_id,
                },
            )
        )
    return docs


def save_jsonl(docs: list[Document], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            row = {"page_content": doc.page_content, "metadata": doc.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    all_docs = build_chunks()
    save_jsonl(all_docs, OUT_JSONL)

    # ---- 验收 ----
    print(f"总块数: {len(all_docs)}")
    for doc in all_docs:
        if doc.metadata["店名"] == "大富烧鸡":
            print(f"\n--- 抽查：{doc.metadata['展示名']}（id={doc.metadata['chunk_id']}）---")
            print("metadata:", json.dumps(doc.metadata, ensure_ascii=False))
            print("正文:")
            print(doc.page_content)

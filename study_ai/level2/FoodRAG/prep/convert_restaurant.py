"""converter：restaurant.md → restaurant.json + 质检台对账。

【设计笔记】

门口装质检台（垃圾别进厨房）：
    md 是给人看的，json 是给程序看的。转换时顺便对账——
    缺字段、人均不是数字、商圈多值、占位符没换，全部当场报出来。
    全绿才输出 json，有红有黄照样输出但打印警告，由人决定改不改。

解析规则（对着 md 的表格格式）：
    只认以 | 开头的行；第1行是表头（校验用），第2行是 ---- 分隔线（跳过），
    其余每行按 | 切成单元格，strip 掉首尾空格（"大富烧鸡 "→"大富烧鸡"）。
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"  # prep/ 的上一层才是项目根
IN_MD = DATA_DIR / "restaurant.md"
OUT_JSON = DATA_DIR / "restaurant.json"

FIELDS = ["店名", "分店", "品类", "人均", "商圈", "适合场景",
          "招牌菜", "特点评价", "避雷点", "营业时间", "地址", "推荐人"]
# 必填字段：缺了就没法检索/过滤
REQUIRED = ["店名", "品类", "人均"]


def parse_rows() -> list[dict]:
    """md 表格 → dict 列表。"""
    rows: list[dict] = []
    in_table = False
    for line_no, line in enumerate(IN_MD.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not in_table:
            in_table = True
            if cells != FIELDS:
                raise ValueError(f"第{line_no}行表头和字段表对不上：{cells}")
            continue  # 表头行不产数据
        if all(set(c) <= {"-", " "} for c in cells):
            continue  # ---- 分隔线，跳过
        if len(cells) != len(FIELDS):
            raise ValueError(f"第{line_no}行列数 {len(cells)} != {len(FIELDS)}，检查竖线数量")
        rows.append(dict(zip(FIELDS, cells)))
    return rows


def validate(rows: list[dict]) -> list[str]:
    """质检台：返回警告列表（空 = 全绿）。"""
    warnings = []
    for i, row in enumerate(rows, start=1):
        name = row["店名"] or f"(第{i}行)"
        # ① 必填字段
        for field in REQUIRED:
            if not row[field]:
                warnings.append(f"[{name}] 必填字段「{field}」是空的")
        # ② 人均必须是纯数字（Chroma 数值过滤的根基）
        price = row["人均"]
        if price and not price.isdigit():
            warnings.append(f"[{name}] 人均「{price}」不是纯数字，数值过滤会失效")
        # ③ 商圈必须单值（Chroma 不支持列表，多值=过滤废掉）
        if any(sep in row["商圈"] for sep in ("、", "，", ",")):
            warnings.append(f"[{name}] 商圈「{row['商圈']}」有多个值，违反单值规则")
        # ④ 占位符没换（值等于字段名本身 = 填表时忘了替换）
        for field in FIELDS:
            if row[field] == field:
                warnings.append(f"[{name}] 「{field}」还是占位符，忘了填真数据")
    return warnings


def main() -> None:
    rows = parse_rows()

    # ---- 对账报告 ----
    print(f"解析出 {len(rows)} 家店（一分店一行）")
    warnings = validate(rows)
    if warnings:
        print(f"\n质检警告 {len(warnings)} 条：")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("质检全绿 ✅")

    # ---- 输出 json（人均转 int，Chroma 数值过滤需要）----
    for row in rows:
        if row["人均"].isdigit():
            row["人均"] = int(row["人均"])
    OUT_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已输出: {OUT_JSON}")


if __name__ == "__main__":
    main()

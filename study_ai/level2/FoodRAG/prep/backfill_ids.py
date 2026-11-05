"""一次性补号：给两本账本的每行发身份证（菜 c001…、店 s001…）。

【设计笔记】为什么要有 id——名字是门牌，id 是身份证：

    菜/店改名（"拌黄花"→"拌黄花菜"），名字一换，旧答案的来源、块的身份全断线。
    id 一旦发出去就写死在账本里，改名只是换门牌，身份证不变——
    来源追溯、将来的增删改查，都靠它指认"到底是哪道菜/哪家店"。

发号规矩：
    菜品 c001 起、饭店 s001 起，按账本行序发；已有 id 的行不重发（幂等）。
    新菜新店进账（将来的 CRUD 接口）= 接着 max 往下发，永不回收。
    不用内容哈希：人要能口头报"查一下 c017"，哈希没人念得出来。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PLAN = [
    {"json": DATA_DIR / "jinyan_menu.json", "prefix": "c"},
    {"json": DATA_DIR / "restaurant.json", "prefix": "s"},
]


def next_free_number(used: set[int]) -> int:
    """接着用过的最大号往下发（不回收空洞：删了 c003，下一个还是 max+1）。"""
    return (max(used) + 1) if used else 1


def backfill(path: Path, prefix: str) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    pat = re.compile(rf"^{prefix}(\d+)$")
    used = {m.group(1) for r in rows if (m := pat.match(str(r.get("id", ""))))}
    used_nums = {int(n) for n in used}
    added = []
    for row in rows:
        if not pat.match(str(row.get("id", ""))):
            n = next_free_number(used_nums)
            row["id"] = f"{prefix}{n:03d}"
            used_nums.add(n)
            added.append((row["id"], row.get("菜名") or row.get("店名")))
    # id 字段挪到行首：账本翻开先见身份证（dict 保序，弹出再重插）
    rows = [dict({"id": row["id"]}, **{k: v for k, v in row.items() if k != "id"})
            if "id" in row else row for row in rows]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{path.name}: 共{len(rows)}行，新发号 {len(added)} 个")
    for i, name in added:
        print(f"  {i} ← {name}")


if __name__ == "__main__":
    for plan in PLAN:
        backfill(plan["json"], plan["prefix"])

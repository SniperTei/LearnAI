"""lost_update_demo.py — 站 0 教学实验：JSON 账本的两大隐患立案（非 bug）

【学徒注】这不是修 bug，是故意让毛病暴露给用户看——每站先见病灶再上新药。
两案都在临时目录的账本副本上案发，真账一字不动。第 1 站换 SQLite 用事务结案。

案情一【丢更新】：两个收货员同时收两张单（1001 收 200、1003 收 100，
  都是 A003）。两人各自遵守"读一份→改这份→存这份"的单人纪律，但交错执行：
  都在对方落盘前读走了旧账——后存的把先存的冲掉。
  对账：期望 46+200+100=346，实际 246（乙的 100 凭空蒸发）。

案情二【半写】：receive_goods 要写两次文件（先订单账后库存账）。
  子进程写完第一本账就 os._exit(1) 模拟中途断电——注意 os._exit 连
  finally 和 atexit 都不跑，这是最接近"进程猝死"的手段。
  重启查账：订单"已收货"了、库存却没涨——两本账对不上。

用法：python scripts/lost_update_demo.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"


def jread(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def jwrite(path: Path, data: list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


# ---------------------------- 案情一：丢更新 ----------------------------

def demo_lost_update(tmp: Path) -> None:
    print("案情一【丢更新】：两个收货员并发收 A003")
    print("背景：A003 库存 46；甲收 PO-2026-1001（+200），乙收 PO-2026-1003（+100）")
    print()

    inv_file = tmp / "inventory.json"

    def timeline(who: str, msg: str) -> None:
        print(f"  T{timeline.n} [{who}] {msg}")
        timeline.n += 1
    timeline.n = 1

    # 甲乙各自"读一份"——都在对方存盘前读走了同一版旧账（46）
    inv_a = jread(inv_file)
    timeline("甲", "读库存：A003 = 46（准备 +200）")
    inv_b = jread(inv_file)
    timeline("乙", "读库存：A003 = 46（准备 +100）")

    # 乙先改先存
    row_b = next(r for r in inv_b if r["sku"] == "A003")
    row_b["quantity"] += 100
    jwrite(inv_file, inv_b)
    timeline("乙", "存库存：46 + 100 = 146 ✓ 落盘")

    # 甲后改后存——但他手里是从头到尾没刷新过的旧照片（46）
    row_a = next(r for r in inv_a if r["sku"] == "A003")
    row_a["quantity"] += 200
    jwrite(inv_file, inv_a)
    timeline("甲", "存库存：46 + 200 = 246 ✓ 落盘（整份覆盖，乙的 100 被冲掉）")

    final = next(r for r in jread(inv_file) if r["sku"] == "A003")["quantity"]
    print()
    print(f"  对账：期望 46 + 200 + 100 = 346，实际 = {final}")
    print("  结论：两人都没犯规（各自都是读→改→存），输在「读」和「存」之间")
    print("        隔着一个别人。后写的整份覆盖 = 先写的更新丢失。")
    print("  第 1 站药方：SQLite 事务 + BEGIN IMMEDIATE 写锁——")
    print("        甲拿到锁后乙的读会等他提交，乙读到的是 246 而不是 46。")
    print()


# ---------------------------- 案情二：半写 ----------------------------

def crash_child(tmp: Path) -> None:
    """子进程：走 receive_goods 的前半段——写完订单账就猝死（不写库存账）"""
    orders = jread(tmp / "orders.json")
    for o in orders:
        if o["order_id"] == "PO-2026-1001":
            o["status"] = "已收货"
            o["received_at"] = date.today().isoformat()
    jwrite(tmp / "orders.json", orders)
    print("  [子进程] 写①完成：PO-2026-1001 → 已收货（订单账落盘）")
    print("  [子进程] !! 断电模拟：写②（库存账）之前 os._exit(1)，")
    print("  [子进程]    finally / atexit 全不跑——最接近进程猝死")
    sys.stdout.flush()
    os._exit(1)


def demo_half_write(tmp: Path) -> None:
    print("案情二【半写】：收货写两本账，写完第一本就崩")
    print("背景：receive_goods 合同 = ①订单账改「已收货」+ ②库存账加数量，")
    print("      两写之间崩 = 半写状态。")
    print()

    # 子进程案发（真进程猝死，不是 try/except 演戏）
    r = subprocess.run([sys.executable, str(Path(__file__).absolute()),
                        "--crash-child", str(tmp)],
                       capture_output=True, text=True)
    print(r.stdout, end="")

    # 重启后查账
    order = next(o for o in jread(tmp / "orders.json")
                 if o["order_id"] == "PO-2026-1001")
    stock = next(x["quantity"] for x in jread(tmp / "inventory.json")
                 if x["sku"] == "A003")
    print(f"  [重启后查账] 订单 PO-2026-1001 状态 = {order['status']}（写了①）")
    print(f"  [重启后查账] A003 库存 = {stock}（②没跑到，还是 46）")
    print()
    print("  对账：货签收了 200 个，账上却查无此货——两本账对不上，")
    print("        月底盘点时才会炸出 200 个轴承的差额。")
    print("  第 1 站药方：两写包进同一个 SQLite 事务——commit 之前崩，")
    print("        WAL 会自动回滚，两本账一起退回没收货的样子（要么全成要么全滚）。")
    print()


def main() -> None:
    # 案情二的子进程入口：python lost_update_demo.py --crash-child <tmp>
    if "--crash-child" in sys.argv:
        crash_child(Path(sys.argv[sys.argv.index("--crash-child") + 1]))
        return

    tmp = Path(tempfile.mkdtemp(prefix="supply_demo_"))
    try:
        for fname in ("inventory.json", "orders.json"):
            shutil.copy(DATA / fname, tmp / fname)
        demo_lost_update(tmp)

        # 案情二重置副本后再案发
        for fname in ("inventory.json", "orders.json"):
            shutil.copy(DATA / fname, tmp / fname)
        demo_half_write(tmp)

        print("两案立案完毕。真账一字未动（全程跑在临时副本上）。")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

"""eval.py — supply_agent 回归考卷（不烧 LLM，工具函数级直调）

【学徒注】学 FoodRAG eval.py 的三条纪律：
1. 判卷不烧 LLM——考的是工具函数的确定性输出（查询/闸门/写账）。
   LLM 会不会路由到正确的工具，那是六连测手跑的事，不进考卷。
2. 数据隔离——考卷跑在临时目录的账本副本上，跑完真账一字未动。
   手法：monkeypatch 各 Server 模块的路径常量指向临时副本，finally 还原。
   （站 1 换 SQLite 后，隔离钥匙换成环境变量 SUPPLY_DB_PATH，断言不动。）
3. 每个真实 bug 进考卷当回归锚点——改完代码必跑 eval。

考卷结构（17 题）：
  查询组 ①~⑨：六连测的函数级等价（命中/大小写/诚实拒答/排序/别名）
  闸门组 ⑩~⑬：三个写工具的 confirmed 闸门 + 假数据拒单（账本必须零变化）
  写账组 ⑭~⑰：真实写账（下单/取消幂等/收货全链 46→246→346/状态边界）

用法：python eval.py   （全绿 exit 0，有挂科 exit 1）
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "servers"))

import inventory_server as inv      # noqa: E402
import logistics_server as log      # noqa: E402
import procurement_server as proc   # noqa: E402
import supplier_server as sup       # noqa: E402

DATA_DIR = BASE / "data"
LEDGERS = ["inventory.json", "orders.json", "suppliers.json", "logistics.json"]

# 隔离钥匙：各 Server 模块里所有会碰账本的路径常量，全部改指临时副本
PATCH_TARGETS = [
    (inv, "INVENTORY_FILE", "inventory.json"),
    (inv, "ORDERS_FILE", "orders.json"),
    (sup, "DATA_FILE", "suppliers.json"),
    (log, "DATA_FILE", "logistics.json"),
    (proc, "ORDERS_FILE", "orders.json"),
    (proc, "SUPPLIERS_FILE", "suppliers.json"),
    (proc, "INVENTORY_FILE", "inventory.json"),
]

FAILURES = []


def patch_paths(tmp: Path) -> dict:
    """把各模块路径常量指到临时副本，返回原值备还原"""
    originals = {}
    for mod, attr, fname in PATCH_TARGETS:
        originals[(mod, attr)] = getattr(mod, attr)
        setattr(mod, attr, tmp / fname)
    return originals


def restore_paths(originals: dict) -> None:
    for (mod, attr), value in originals.items():
        setattr(mod, attr, value)


def reset_ledgers(tmp: Path) -> None:
    """把种子账本重新拷进临时目录（写账大题前重置，题与题互不串状态）"""
    for fname in LEDGERS:
        shutil.copy(DATA_DIR / fname, tmp / fname)


def jread(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def check(n: int, name: str, cond: bool, detail: str = "") -> None:
    line = f"【测{n:02d}】{name} {'PASS' if cond else 'FAIL'}"
    if not cond and detail:
        line += f"  ← {detail}"
    print(line)
    if not cond:
        FAILURES.append(n)


# ---------------------------- 查询组 ----------------------------

def test_query_group(tmp: Path):
    recs = json.loads(inv.query_stock("A003"))
    check(1, "query_stock 精确命中", len(recs) == 1
          and recs[0]["quantity"] == 46 and recs[0]["safety_stock"] == 100,
          f"got {recs}")

    recs = json.loads(inv.query_stock("a003"))
    check(2, "SKU 大小写归一", len(recs) == 1 and recs[0]["sku"] == "A003")

    r = inv.query_stock("X999")
    check(3, "查不到诚实返回+SKU清单", r.startswith("未找到 SKU X999") and "A003" in r,
          f"got {r!r}")

    low = json.loads(inv.list_low_stock())
    skus = [x["sku"] for x in low]
    # 缺口 = quantity - safety_stock：C002(-80) < A003(-54) < A005(-12)
    check(4, "低库存清单按缺口排序", skus == ["C002", "A003", "A005"], f"got {skus}")

    hits = json.loads(sup.search_suppliers("轴承"))
    ids = [h["id"] for h in hits]
    check(5, "品类搜索+交期排序", ids == ["SUP-001", "SUP-002"]
          and hits[0]["lead_time_days"] == 7 and hits[1]["lead_time_days"] == 12,
          f"got {ids}")

    hits = json.loads(sup.search_suppliers("华精"))
    check(6, "别名命中归一", len(hits) == 1 and hits[0]["id"] == "SUP-001",
          f"got {[h['id'] for h in hits]}")

    r = sup.search_suppliers("量子纠缠材料")
    check(7, "假词诚实返回+品类菜单", "未找到" in r and "轴承" in r, f"got {r!r}")

    t = json.loads(log.query_logistics("LG-2026-0001"))
    check(8, "运单详情+事件流", t["status"] == "在途" and len(t["events"]) == 3)

    lst = json.loads(log.list_in_transit())
    nos = [x["tracking_no"] for x in lst]
    check(9, "在途清单按eta排序", nos == ["LG-2026-0002", "LG-2026-0003", "LG-2026-0001"],
          f"got {nos}")


# ---------------------------- 闸门组 ----------------------------

def test_gate_group(tmp: Path):
    # ⑩ 下单闸门：拒写 + 账本行数不变
    before = len(jread(tmp / "orders.json"))
    r = proc.create_order("SUP-001", "A003", 100, confirmed=False)
    after = len(jread(tmp / "orders.json"))
    check(10, "create_order 闸门", "未确认" in r and after == before,
          f"r={r!r} rows {before}->{after}")

    # ⑪ 假供应商：confirmed=true 也拒（先验证后写账的铁律）
    r = proc.create_order("SUP-999", "A003", 100, confirmed=True)
    still = len(jread(tmp / "orders.json"))
    check(11, "假供应商拒单", "未找到供应商" in r and still == before,
          f"r={r!r} rows {before}->{still}")

    # ⑫ 取消闸门
    r = proc.cancel_order("PO-2026-1001", confirmed=False)
    o = next(x for x in jread(tmp / "orders.json") if x["order_id"] == "PO-2026-1001")
    check(12, "cancel_order 闸门", "未确认" in r and o["status"] == "已下单",
          f"r={r!r} status={o['status']}")

    # ⑬ 收货闸门：拒写 + 两本账都零变化
    orders_before = jread(tmp / "orders.json")
    inv_before = jread(tmp / "inventory.json")
    r = inv.receive_goods("PO-2026-1001", confirmed=False)
    check(13, "receive_goods 闸门", "未确认" in r
          and jread(tmp / "orders.json") == orders_before
          and jread(tmp / "inventory.json") == inv_before,
          f"r={r!r}")


# ---------------------------- 写账组 ----------------------------

def test_write_group(tmp: Path):
    # ⑭ 真实下单：种子最大号 1002 → 新单 1003
    r = proc.create_order("SUP-001", "A003", 100, confirmed=True)
    o = json.loads(r)
    in_file = [x for x in jread(tmp / "orders.json")
               if x["order_id"] == "PO-2026-1003"]
    check(14, "真实下单入账", o["order_id"] == "PO-2026-1003"
          and o["status"] == "已下单" and len(in_file) == 1
          and in_file[0]["sku"] == "A003" and in_file[0]["quantity"] == 100,
          f"r={r!r}")

    # ⑮ 取消 + 幂等
    r = proc.cancel_order("PO-2026-1002", confirmed=True)
    o = json.loads(r)
    r2 = proc.cancel_order("PO-2026-1002", confirmed=True)
    check(15, "取消订单+再取消幂等", o["status"] == "已取消"
          and "已经是已取消" in r2, f"r={r!r} r2={r2!r}")

    # ⑯ 收货全链（重置账本，本题自成一局）：
    #    下单 1003(100×A003) → 收货 1001(200) → A003=246 → 收货 1003 → 346
    reset_ledgers(tmp)
    proc.create_order("SUP-001", "A003", 100, confirmed=True)  # PO-2026-1003
    r1 = json.loads(inv.receive_goods("PO-2026-1001", confirmed=True))
    r2 = json.loads(inv.receive_goods("PO-2026-1003", confirmed=True))
    orders = {x["order_id"]: x for x in jread(tmp / "orders.json")}
    stock = next(x["quantity"] for x in jread(tmp / "inventory.json")
                 if x["sku"] == "A003")
    check(16, "收货全链 46→246→346",
          r1["stock_after"] == 246 and r2["stock_after"] == 346 and stock == 346
          and orders["PO-2026-1001"]["status"] == "已收货"
          and orders["PO-2026-1003"]["status"] == "已收货"
          and orders["PO-2026-1001"].get("received_at")
          and orders["PO-2026-1003"].get("received_at"),
          f"r1={r1} r2={r2} stock={stock}")

    # ⑰ 收货边界：已取消的单拒收、不存在的单诚实说没有（重置后再造一单取消单）
    reset_ledgers(tmp)
    proc.cancel_order("PO-2026-1001", confirmed=True)
    r_cancelled = inv.receive_goods("PO-2026-1001", confirmed=True)
    r_missing = inv.receive_goods("PO-2026-9999", confirmed=True)
    stock = next(x["quantity"] for x in jread(tmp / "inventory.json")
                 if x["sku"] == "A003")
    check(17, "已取消拒收+不存在单诚实", "才能收货" in r_cancelled
          and "未找到订单" in r_missing and stock == 46,
          f"r_cancelled={r_cancelled!r} r_missing={r_missing!r} stock={stock}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="supply_eval_"))
    originals = patch_paths(tmp)
    try:
        reset_ledgers(tmp)
        test_query_group(tmp)
        test_gate_group(tmp)
        test_write_group(tmp)
    finally:
        restore_paths(originals)
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 50)
    if FAILURES:
        print(f"结果：{len(FAILURES)} 题挂科：{FAILURES}")
        return 1
    print("结果：17 题全绿 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

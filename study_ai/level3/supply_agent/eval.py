"""eval.py — supply_agent 回归考卷（不烧 LLM，工具函数级直调）

【学徒注】学 FoodRAG eval.py 的三条纪律：
1. 判卷不烧 LLM——考的是工具函数的确定性输出（查询/闸门/写账）。
   LLM 会不会路由到正确的工具，那是六连测手跑的事，不进考卷。
2. 数据隔离——考卷跑在临时目录的账本副本上，跑完真账一字未动。
   手法：monkeypatch 各 Server 模块的路径常量指向临时副本，finally 还原。
3. 每个真实 bug 进考卷当回归锚点——改完代码必跑 eval。
   （L1 修的真 bug：采购取消"已收货"单曾被放行 → 测26 锚定）

考卷结构（26 题）：
  查询组 ①~⑨：六连测的函数级等价（命中/大小写/诚实拒答/排序/别名）
  闸门组 ⑩~⑬：写工具的 confirmed 闸门 + 假数据拒单（账本必须零变化）
  写账组 ⑭~⑰：真实写账（下单/取消幂等/收货全链 46→246→346/状态边界）
  卖出组 ⑱~㉖：客户搜索 / 销售闸门 / 出库闸门 / 卖出+出库扣减 /
               库存不足拒 / 双向闭环 / 取消幂等+已收货拒取消

用法：python eval.py   （全绿 exit 0，有挂科 exit 1）
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "servers"))

import customer_server as cust      # noqa: E402
import inventory_server as inv      # noqa: E402
import logistics_server as log      # noqa: E402
import procurement_server as proc   # noqa: E402
import sales_server as sales        # noqa: E402
import supplier_server as sup       # noqa: E402

DATA_DIR = BASE / "data"
LEDGERS = ["inventory.json", "orders.json", "suppliers.json", "logistics.json",
           "customers.json", "sales_orders.json"]

# 隔离钥匙：各 Server 模块里所有会碰账本的路径常量，全部改指临时副本
PATCH_TARGETS = [
    (inv, "INVENTORY_FILE", "inventory.json"),
    (inv, "ORDERS_FILE", "orders.json"),
    (inv, "SALES_ORDERS_FILE", "sales_orders.json"),
    (sup, "DATA_FILE", "suppliers.json"),
    (log, "DATA_FILE", "logistics.json"),
    (proc, "ORDERS_FILE", "orders.json"),
    (proc, "SUPPLIERS_FILE", "suppliers.json"),
    (proc, "INVENTORY_FILE", "inventory.json"),
    (cust, "DATA_FILE", "customers.json"),
    (sales, "SALES_ORDERS_FILE", "sales_orders.json"),
    (sales, "CUSTOMERS_FILE", "customers.json"),
    (sales, "INVENTORY_FILE", "inventory.json"),
]

FAILURES = []
CHECKS = 0


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
    global CHECKS
    CHECKS += 1
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


# ---------------------------- 卖出组 ----------------------------

def test_sales_group(tmp: Path):
    # ⑱ 客户品类搜索：轴承 → 宝安五金 + 福田贸易，按编号稳定排序
    hits = json.loads(cust.search_customers("轴承"))
    ids = [h["id"] for h in hits]
    check(18, "客户品类搜索", ids == ["CUS-002", "CUS-004"], f"got {ids}")

    hits = json.loads(cust.search_customers("福田贸易"))
    check(19, "客户别名命中", len(hits) == 1 and hits[0]["id"] == "CUS-004",
          f"got {[h['id'] for h in hits]}")

    r = cust.search_customers("量子纠缠")
    check(20, "客户假词诚实+品类菜单", "未找到" in r and "深沟球轴承" in r,
          f"got {r!r}")

    # ㉑ 销售闸门：拒写 + 账本行数不变
    before = len(jread(tmp / "sales_orders.json"))
    r = sales.create_sales_order("CUS-002", "A004", 200, confirmed=False)
    check(21, "create_sales_order 闸门", "未确认" in r
          and len(jread(tmp / "sales_orders.json")) == before, f"r={r!r}")

    # ㉒ 出库闸门：拒写 + 两本账零变化
    so_before = jread(tmp / "sales_orders.json")
    inv_before = jread(tmp / "inventory.json")
    r = inv.outbound_goods("SO-2026-2001", confirmed=False)
    check(22, "outbound_goods 闸门", "未确认" in r
          and jread(tmp / "sales_orders.json") == so_before
          and jread(tmp / "inventory.json") == inv_before, f"r={r!r}")

    # ㉓ 真实卖出 + 出库扣减（重置，本题自成一局）：
    #    开单 SO-2026-2003(200×A004) → 出库种子单 SO-2001(500×A004)
    #    → A004 2600-500=2100，2001 变已发货
    reset_ledgers(tmp)
    o = json.loads(sales.create_sales_order("CUS-002", "A004", 200, confirmed=True))
    r2 = json.loads(inv.outbound_goods("SO-2026-2001", confirmed=True))
    stock = next(x["quantity"] for x in jread(tmp / "inventory.json")
                 if x["sku"] == "A004")
    so2001 = next(x for x in jread(tmp / "sales_orders.json")
                  if x["order_id"] == "SO-2026-2001")
    check(23, "卖出入账+出库扣减", o["order_id"] == "SO-2026-2003"
          and o["status"] == "已下单" and r2["stock_after"] == 2100
          and stock == 2100 and so2001["status"] == "已发货"
          and so2001.get("shipped_at"),
          f"o={o} r2={r2} stock={stock}")

    # ㉔ 库存不足拒出库（紧接上局：D003=60，开 99999 的大单）
    sales.create_sales_order("CUS-001", "D003", 99999, confirmed=True)  # SO-2026-2004
    r = inv.outbound_goods("SO-2026-2004", confirmed=True)
    d003 = next(x["quantity"] for x in jread(tmp / "inventory.json")
                if x["sku"] == "D003")
    so2004 = next(x for x in jread(tmp / "sales_orders.json")
                  if x["order_id"] == "SO-2026-2004")
    check(24, "库存不足拒出库", "库存不足" in r and d003 == 60
          and so2004["status"] == "已下单", f"r={r!r} d003={d003}")

    # ㉕ 双向闭环（重置）：收货 1001(A003 46→246) → 卖 50 → 出库 → 196
    reset_ledgers(tmp)
    inv.receive_goods("PO-2026-1001", confirmed=True)
    sales.create_sales_order("CUS-002", "A003", 50, confirmed=True)  # SO-2026-2003
    r = json.loads(inv.outbound_goods("SO-2026-2003", confirmed=True))
    stock = next(x["quantity"] for x in jread(tmp / "inventory.json")
                 if x["sku"] == "A003")
    check(25, "双向闭环 收→卖→出 246→196", r["stock_after"] == 196
          and stock == 196, f"r={r} stock={stock}")

    # ㉖ 取消销售单幂等 + 采购侧已收货拒取消（L1 修的真 bug 锚点）
    reset_ledgers(tmp)
    o = json.loads(sales.cancel_sales_order("SO-2026-2002", confirmed=True))
    r2 = sales.cancel_sales_order("SO-2026-2002", confirmed=True)
    inv.receive_goods("PO-2026-1001", confirmed=True)  # 1001 → 已收货
    r3 = proc.cancel_order("PO-2026-1001", confirmed=True)
    check(26, "取消幂等+已收货拒取消", o["status"] == "已取消"
          and "已经是已取消" in r2 and "才能取消" in r3,
          f"o={o} r2={r2!r} r3={r3!r}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="supply_eval_"))
    originals = patch_paths(tmp)
    try:
        reset_ledgers(tmp)
        test_query_group(tmp)
        test_gate_group(tmp)
        test_write_group(tmp)
        test_sales_group(tmp)
    finally:
        restore_paths(originals)
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 50)
    if FAILURES:
        print(f"结果：{len(FAILURES)} 题挂科：{FAILURES}")
        return 1
    print(f"结果：{CHECKS} 题全绿 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

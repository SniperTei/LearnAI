"""销售 MCP Server（卖手侧的"写" Server，stdio 传输）

【学徒注】与 procurement_server 完全对称（买手侧：supplier 情报 +
procurement 写账；卖手侧：customer 情报 + sales 写账）：

  - create_sales_order(customer_id, sku, quantity, confirmed)  开销售单（写账）
  - get_sales_order(order_id)                                  查单笔销售单（读）
  - cancel_sales_order(order_id, confirmed)                    取消销售单（写账）

教学要点：
1. 同款确认闸门 + 同款状态机（已下单 → 已取消 / 已发货）——
   好的合同抄过来就用，这就是"合同住 docstring"的好处：可复制。
2. 【下单不查库存】先接单后备货：接单时库存够不算数（可能明天就被别的
   单出光了），库存够不够是出库时（inventory 的 outbound_goods）的事——
   和采购对称：下单不问供应商有没有货，收货那天才知道。
3. 写前先验证客户和 SKU 真实存在（防幻觉进账本）——同 procurement 铁律。
4. 已发货的单不能取消（货都走了，取消要走退货流程，本系统不装懂）。
"""
import json
from datetime import date
from pathlib import Path

from fastmcp import FastMCP

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SALES_ORDERS_FILE = DATA_DIR / "sales_orders.json"
CUSTOMERS_FILE = DATA_DIR / "customers.json"
INVENTORY_FILE = DATA_DIR / "inventory.json"

mcp = FastMCP("sales")


def _read(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _next_order_id(orders: list) -> str:
    """发号器：SO-2026-2001 式流水号（学 procurement 的 PO 发号思想）"""
    max_no = max((int(o["order_id"].split("-")[-1]) for o in orders), default=2000)
    return f"SO-2026-{max_no + 1}"


@mcp.tool()
def create_sales_order(customer_id: str, sku: str, quantity: int,
                       confirmed: bool = False) -> str:
    """向客户开销售订单（写操作，会写入 sales_orders.json）。
    参数：customer_id 客户编号（如 CUS-002）、sku 物料编号（如 A004）、
    quantity 数量、confirmed 是否已取得用户明确同意。
    【重要合同】confirmed=false 时本工具不会写账，只会返回提示——
    必须先向用户复述：卖给哪家客户、什么物料、多少数量，取得用户明确
    同意后，再带 confirmed=true 重新调用本工具，销售单才会真正创建。
    """
    if not confirmed:
        # 【学徒注】确认闸门，与 create_order 同款合同
        return ("销售单未确认：请先向用户复述订单要素——卖给哪家客户"
                f"（{customer_id}）、什么物料（{sku}）、数量多少（{quantity}），"
                "取得用户明确同意后，再带 confirmed=true 重新调用。")

    # 先验证客户和 SKU 真实存在（见模块注释第 3 条）
    cust = next((c for c in _read(CUSTOMERS_FILE)
                 if c["id"].upper() == customer_id.upper()), None)
    if cust is None:
        return f"开单失败：未找到客户 {customer_id}。"
    item = next((r for r in _read(INVENTORY_FILE)
                 if r["sku"].upper() == sku.upper()), None)
    if item is None:
        return f"开单失败：未找到 SKU {sku}。"
    if quantity <= 0:
        return "开单失败：数量必须大于 0。"

    # 写账纪律：读一份 → 改这份 → 存这份
    orders = _read(SALES_ORDERS_FILE)
    order = {
        "order_id": _next_order_id(orders),
        "customer_id": cust["id"],
        "sku": item["sku"],
        "quantity": quantity,
        "status": "已下单",
        "created_at": date.today().isoformat(),
        "note": f"由供小链 Agent 创建：向 {cust['name']} 销售 {item['name']}",
    }
    orders.append(order)
    _write(SALES_ORDERS_FILE, orders)
    return json.dumps(order, ensure_ascii=False)


@mcp.tool()
def get_sales_order(order_id: str) -> str:
    """按销售单号（如 SO-2026-2001）查询单笔销售订单的详情。
    """
    for o in _read(SALES_ORDERS_FILE):
        if o["order_id"].upper() == order_id.upper():
            return json.dumps(o, ensure_ascii=False)
    return f"未找到销售单 {order_id}。"


@mcp.tool()
def cancel_sales_order(order_id: str, confirmed: bool = False) -> str:
    """取消销售订单（写操作）。参数：order_id 销售单号、confirmed 是否已取得
    用户明确同意。
    【重要合同】与 create_sales_order 相同：confirmed=false 时不写账，必须先向
    用户复述要取消的销售单信息并取得明确同意，再带 confirmed=true 重新调用。
    只有"已下单"状态的销售单能取消；已发货的单无法取消。
    """
    if not confirmed:
        return (f"取消未确认：请先向用户复述要取消的销售单（{order_id}），"
                "取得用户明确同意后，再带 confirmed=true 重新调用。")

    orders = _read(SALES_ORDERS_FILE)     # 读一份
    for o in orders:                      # 改这份
        if o["order_id"].upper() == order_id.upper():
            if o["status"] == "已取消":
                return f"销售单 {o['order_id']} 已经是已取消状态，无需重复操作。"
            if o["status"] != "已下单":
                return (f"销售单 {o['order_id']} 当前状态为「{o['status']}」，"
                        "只有已下单的销售单才能取消。")
            o["status"] = "已取消"
            _write(SALES_ORDERS_FILE, orders)  # 存这份
            return json.dumps(o, ensure_ascii=False)
    return f"未找到销售单 {order_id}，无法取消。"


if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        from http_entry import run_http
        run_http(mcp, default_port=8106)  # 端口与 agent.py 的 SERVER_PORTS 对齐
    else:
        mcp.run(show_banner=False, log_level="ERROR")  # stdio 默认：agent 子进程；stderr留干净

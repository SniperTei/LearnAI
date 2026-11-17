"""采购 MCP Server（本项目唯一的"写" Server，stdio 传输）

工具：
  - create_order(supplier_id, sku, quantity, confirmed)  下采购单（写账）
  - get_order(order_id)                                   查单笔订单（读）
  - cancel_order(order_id, confirmed)                     取消订单（写账）

教学要点：
1. 【确认闸门】写工具的安全合同住在这里，不住 prompt：
   confirmed=False 时绝不写账，直接返回"请先向用户复述订单要素并取得
   明确同意"。LLM 读了工具返回值自然学会守规矩——比在 system prompt 里
   求它"请不要直接下单"可靠得多（prompt 是请求，返回值是铁板）。
2. 【写账纪律】学 FoodRAG meals CRUD 的教训：读一份 → 改这份 → 存这份，
   中间绝不重读文件（防止读到别人改过的版本把自己的改动冲掉）。
3. 下单前先验证供应商和 SKU 真实存在——不验证就写账等于允许 LLM 编造
   供应商下单（幻觉直接进账本）。
"""
import json
from datetime import date
from pathlib import Path

from fastmcp import FastMCP

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ORDERS_FILE = DATA_DIR / "orders.json"
SUPPLIERS_FILE = DATA_DIR / "suppliers.json"
INVENTORY_FILE = DATA_DIR / "inventory.json"

mcp = FastMCP("procurement")


def _read(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _next_order_id(orders: list) -> str:
    """发号器：PO-2026-1001 式流水号，取现有最大号 +1（学 backfill_ids 发号思想）"""
    max_no = max((int(o["order_id"].split("-")[-1]) for o in orders), default=1000)
    return f"PO-2026-{max_no + 1}"


@mcp.tool()
def create_order(supplier_id: str, sku: str, quantity: int,
                 confirmed: bool = False) -> str:
    """向供应商下采购订单（写操作，会写入 orders.json）。
    参数：supplier_id 供应商编号（如 SUP-001）、sku 物料编号（如 A003）、
    quantity 数量、confirmed 是否已取得用户明确同意。
    【重要合同】confirmed=false 时本工具不会写账，只会返回提示——
    必须先向用户复述：向哪家供应商、买什么物料、买多少，取得用户明确
    同意后，再带 confirmed=true 重新调用本工具，订单才会真正创建。
    """
    if not confirmed:
        # 【学徒注】确认闸门本体。不写账，把"该怎么补手续"说清楚，
        # LLM 拿到这段话会转述给用户走确认流程
        return ("订单未确认：请先向用户复述订单要素——向哪家供应商"
                f"（{supplier_id}）、买什么物料（{sku}）、数量多少（{quantity}），"
                "取得用户明确同意后，再带 confirmed=true 重新调用。")

    # 先验证供应商和 SKU 真实存在（见模块注释第 3 条）
    sup = next((s for s in _read(SUPPLIERS_FILE)
                if s["id"].upper() == supplier_id.upper()), None)
    if sup is None:
        return f"下单失败：未找到供应商 {supplier_id}。"
    item = next((r for r in _read(INVENTORY_FILE)
                 if r["sku"].upper() == sku.upper()), None)
    if item is None:
        return f"下单失败：未找到 SKU {sku}。"
    if quantity <= 0:
        return "下单失败：数量必须大于 0。"

    # 写账纪律：读一份 → 改这份 → 存这份（见模块注释第 2 条）
    orders = _read(ORDERS_FILE)
    order = {
        "order_id": _next_order_id(orders),
        "supplier_id": sup["id"],
        "sku": item["sku"],
        "quantity": quantity,
        "status": "已下单",
        "created_at": date.today().isoformat(),
        "note": f"由供小链 Agent 创建：向 {sup['name']} 采购 {item['name']}",
    }
    orders.append(order)
    _write(ORDERS_FILE, orders)
    return json.dumps(order, ensure_ascii=False)


@mcp.tool()
def get_order(order_id: str) -> str:
    """按订单号（如 PO-2026-1001）查询单笔采购订单的详情。
    """
    for o in _read(ORDERS_FILE):
        if o["order_id"].upper() == order_id.upper():
            return json.dumps(o, ensure_ascii=False)
    return f"未找到订单 {order_id}。"


@mcp.tool()
def cancel_order(order_id: str, confirmed: bool = False) -> str:
    """取消采购订单（写操作）。参数：order_id 订单号、confirmed 是否已取得
    用户明确同意。
    【重要合同】与 create_order 相同：confirmed=false 时不写账，必须先向
    用户复述要取消的订单信息并取得明确同意，再带 confirmed=true 重新调用。
    只有"已下单"状态的订单能取消。
    """
    if not confirmed:
        return (f"取消未确认：请先向用户复述要取消的订单（{order_id}），"
                "取得用户明确同意后，再带 confirmed=true 重新调用。")

    orders = _read(ORDERS_FILE)          # 读一份
    for o in orders:                     # 改这份
        if o["order_id"].upper() == order_id.upper():
            if o["status"] == "已取消":
                return f"订单 {o['order_id']} 已经是已取消状态，无需重复操作。"
            # 【学徒注】L1 修的真 bug：站 0 给订单长出"已收货"第三态后，
            # 这里漏了拦——收了的货再"取消"，库存加过的数量就对不上账了。
            # docstring 早就写着"只有已下单的订单能取消"，代码现在才跟上
            if o["status"] != "已下单":
                return (f"订单 {o['order_id']} 当前状态为「{o['status']}」，"
                        "只有已下单的订单才能取消。")
            o["status"] = "已取消"
            _write(ORDERS_FILE, orders)  # 存这份
            return json.dumps(o, ensure_ascii=False)
    return f"未找到订单 {order_id}，无法取消。"


if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        from http_entry import run_http
        run_http(mcp, default_port=8104)  # 端口与 agent.py 的 SERVER_PORTS 对齐
    else:
        mcp.run(show_banner=False, log_level="ERROR")  # stdio 默认：agent 子进程；stderr留干净

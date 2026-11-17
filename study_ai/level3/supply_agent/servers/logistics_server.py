"""物流 MCP Server（只读，stdio 传输）

工具：
  - query_logistics(tracking_no)  查单笔运单（轨迹+状态+ETA）
  - list_in_transit()             列出所有还没签收的运单

教学要点：
1. 轨迹事件按时间正序返回——快递查询的本质是一条事件流，
   数据形状（list of events）天生适合按时间讲清楚。
2. 同样的诚实合同：运单号查不到就明说，不编轨迹。
"""
import json
from pathlib import Path

from fastmcp import FastMCP

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "logistics.json"

mcp = FastMCP("logistics")


def _load() -> list:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


@mcp.tool()
def query_logistics(tracking_no: str) -> str:
    """按运单号（如 LG-2026-0001）查询物流详情：当前状态、起讫地、
    预计到达时间（ETA）和完整轨迹事件。查不到返回"未找到"。
    """
    for t in _load():
        if t["tracking_no"].upper() == tracking_no.upper():
            return json.dumps(t, ensure_ascii=False)
    all_nos = ", ".join(t["tracking_no"] for t in _load())
    return f"未找到运单 {tracking_no}。现有运单：{all_nos}"


@mcp.tool()
def list_in_transit() -> str:
    """列出所有在途（未签收）的运单：含在途和派送中两种状态，
    按预计到达时间从早到晚排序，方便回答"最近到货的是什么"。
    """
    transit = [t for t in _load() if t["status"] != "已签收"]
    if not transit:
        return "当前没有在途运单。"
    transit.sort(key=lambda t: t["eta"])
    # 【学徒注】轨迹 events 全文太长，列表场景只给摘要；单笔详情走 query_logistics
    brief = [{k: t[k] for k in ("tracking_no", "order_ref", "supplier",
                                "origin", "destination", "status", "eta")}
             for t in transit]
    return json.dumps(brief, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(show_banner=False, log_level="ERROR")  # 关横幅和INFO日志，stderr也留干净

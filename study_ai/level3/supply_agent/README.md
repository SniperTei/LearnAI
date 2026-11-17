# supply_agent：供应链智能助手（MCP 版）

从零自己设计的第一个 Agent 项目：通用工厂供应链助手。核心课题——
**MCP 协议本体**（Server 甲方 / Agent 客户端乙方）+ **工具设计谱系实战**（只读 vs 写、确认闸门、无状态代价）。

## 架构图

```
你（终端）
 │
 ▼
agent.py（乙方/客户端，LangChain create_agent + qwen3.7-flash）
 │  经 MCPAdapter 拉起 4 个子进程，走 stdio 管道说 JSON-RPC
 ├── inventory_server.py    库存账   data/inventory.json
 ├── supplier_server.py     供应商账 data/suppliers.json
 ├── logistics_server.py    运单账   data/logistics.json
 └── procurement_server.py  采购单账 data/orders.json   ← 唯一的"写"Server
```

每次你问一句话：LLM 决定调哪把工具 → MCPAdapter 经管道发给对应 Server →
Server 现读账本 → 结果原路返回 → LLM 组织答案。**Agent 从头到尾没碰过
JSON 文件**——数据只能从工具过，这就是 MCP 的甲乙方合同。

## MCP 能力清单（三能力，本项目只用第一个）

| 能力 | 是什么 | 本项目 |
|---|---|---|
| **Tools** | LLM 可调用的函数 | ✅ 9 把（下表） |
| Resources | Server 暴露的只读数据文件 | ❌ 账本走工具返回，够用 |
| Prompts  | Server 预制的提示词模板 | ❌ 人设住 agent.py 的 system prompt |

## 工具清单（读写分离）

| Server | 工具 | 读/写 | 设计要点 |
|---|---|---|---|
| inventory | `query_stock(sku, warehouse="")` / `list_low_stock()` | 只读 | 查不到诚实返回"未找到"+现有SKU清单 |
| supplier | `search_suppliers(keyword)` / `get_supplier(id)` | 只读 | 名称/品类/别名三处匹配；结果按交期排好序 |
| logistics | `query_logistics(tracking_no)` / `list_in_transit()` | 只读 | 单笔给全轨迹；列表给摘要防撑爆上下文 |
| procurement | `create_order(...)` / `get_order(id)` / `cancel_order(...)` | **写** | **确认闸门**（见下） |

## 运行方法

```bash
conda activate study_ai
python agent.py        # 空回车或 Ctrl+C 退出
```

前置：`.env` 里有 `DASHSCOPE_API_KEY`；依赖 `langchain[mcp]`（已装，
含 fastmcp/mcp 运行时）。单独调试某个 Server：`python servers/inventory_server.py`
（stdio 模式，启动后等管道消息，不响不是坏了）。

## 验收六连测（2026-11-17 全绿 ✅）

1. "A003 还有多少？" → query_stock，答 46 个 + 主动提醒低于安全线
2. "哪些东西快用完了" → list_low_stock，列 3 项缺口
3. "谁家供轴承？交期最短的是哪家" → search_suppliers，华精 7 天 vs 恒力 12 天
4. "LG-2026-0001 到哪了" → query_logistics，广州分拨中心 + ETA 11-19
5. "帮我向 SUP-001 下单买 100 个 A003" → 先复述订单要素问确认 → 用户"确认"
   → create_order(confirmed=True) → orders.json 新增 PO-2026-1003 ✅
6. "把刚才那单取消" → 复述订单要素问确认 → cancel_order(confirmed=True)
   → orders.json 状态变已取消 ✅（测完账本已还原）

附加：①"我这边缺料了咋办"（无任何词表关键词）→ 语义直达 list_low_stock，
LLM 语义路由 > FoodRAG 词表路由的活例子；②函数级单测 confirmed=False
两个写工具都拒写。

## 教学沉淀

### 1. 确认闸门：安全约束住工具层，不住 prompt
`confirmed=False` 时不写账，返回"请先复述要素并取得同意"。docstring 就是
合同（LLM 读工具说明懂规矩），返回值就是铁板（忘了规矩也会被弹回来）。
对比 prompt 约束："请不要直接下单"是请求，闸门是物理锁——学 drug_rag
安全链哲学：**约束放在离危险最近的地方**。

### 2. 写账纪律（读改存）
读一份 → 改这份 → 存这份，中间不重读文件。学 FoodRAG meals CRUD 教训
（中途重读会把改动冲掉）。见 procurement_server.py。

### 3. 无状态代价
每次工具调用都重读 JSON：多花磁盘 IO，换来永远读到最新账 + Server 零状态
可随意重启。缓存是优化，不是默认——先用对再求快。

### 4. `command: sys.executable`
MCP Server 是子进程。裸 `"python"` 由 PATH 说了算，容易飘到系统 Python
（那里没装 mcp 包，子进程秒死）。当前解释器的绝对路径才稳。

### 5. 踩坑实录（写代码当天亲测）
- **官方 mcp 2.x 把 FastMCP 改名 MCPServer** → 改用独立的 fastmcp 包
  （`from fastmcp import FastMCP`），`@mcp.tool()` 风格不变
- **fastmcp 依赖抬 starlette 到 1.10 弄坏 fastapi**（FoodRAG web.py 在用）
  → 降回 `starlette<0.51`（0.50.0），两边共存验证通过
- **create_agent 不记对话历史**：用户说"确认"时它失忆 → agent.py 手动累积
  history 每轮全量喂（同 FoodRAG 金鱼记忆一课：模型无状态，账要自己记）
- **stdio 传输时 stdout 是协议通道**，Server 里绝不能 print() 调试

## 谱系对照（这个项目站在哪些前人的肩膀上）

| 设计 | 学的是 |
|---|---|
| JSON 账本 + 发号（PO-2026-1003+1） | FoodRAG 账本风格 / backfill_ids 发号思想 |
| 供应商别名匹配（"华精"→华精轴承制造） | FoodRAG 锁菜别名 |
| 查不到诚实返回"未找到" | FoodRAG 防幻觉 / drug_rag 诚实拒答 |
| 确认闸门住工具返回值 | drug_rag 安全链"约束住离危险最近的地方" |
| LLM 语义路由 vs 词表路由 | FoodRAG router 的反面教材，Agent 的长处 |
| system prompt 人设+规则表 | 陈玉楼算命 agent |

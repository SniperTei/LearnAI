# supply_agent：供应链智能助手（MCP 版，中间商模式）

从零自己设计的第一个 Agent 项目：小型贸易公司（中间商）的供应链助手——
向供应商采购进货、入仓、出库、卖给客户，赚差价。核心课题——
**MCP 协议本体**（Server 甲方 / Agent 客户端乙方）+ **工具设计谱系实战**
（只读 vs 写、确认闸门、无状态代价）+ **按角色挂载的模块化架构**（APP_ROLE）。

> 定位：**学习作业**（JSON 账本 + mock 数据）。学习路线见 [study_plan.md](study_plan.md)；
> 需求背景见 [requirements.md](requirements.md)；真实系统完整方案见
> [supply_plan.md](supply_plan.md)（SQLite / 批次 / 合同 / 报表，另立仓库时参考）。

## 架构图

```
你（终端 或 浏览器 localhost:8000）
 │
 ▼
agent.py / web.py ——两个门脸一个后厨（L3 起）
 │  共用 build_agent() + run_turn()（公共件住 agent.py）
 │  LangChain create_agent + qwen3.7-flash
 │
 │  经 MCPAdapter 按角色挂载六座 MCP Server（挂载即开关，APP_ROLE 说了算）：
 │  传输按座自选（L2）——stdio 子进程（默认）或 HTTP 常驻（--http，Bearer 认证）
 ├── inventory_server.py   库存账（读 + 收货入库/销售出库写，跨账本）
 ├── logistics_server.py   运单账（只读）                ← 两头共用的核心
 ├── supplier_server.py    供应商账（只读）   ┐ 买手侧
 ├── procurement_server.py 采购单账（写，闸门）┘
 ├── customer_server.py    客户账（只读）     ┐ 卖手侧
 └── sales_server.py       销售单账（写，闸门）┘

 Docker（L5）：一个镜像七个容器——六座全 --http + web，
 经 compose 网络互访（服务名即主机名），账本住卷里
```

**APP_ROLE 挂载即开关**（trader 中间商=全挂 / buyer 只买 / seller 只卖）：
不挂载 = LLM 的工具名片里没有 = 它想干也没门。详见"运行方法"。

每次你问一句话：LLM 决定调哪把工具 → MCPAdapter 经管道发给对应 Server →
Server 现读账本 → 结果原路返回 → LLM 组织答案。**Agent 从头到尾没碰过
JSON 文件**——数据只能从工具过，这就是 MCP 的甲乙方合同。

## 一次调用的完整回路：查库存 + 给采购建议

拿验收第 1 测的扩展版走全程：**"A003 还有多少？要不要补点货？"**
分两个阶段：开机挂载（只发生一次）+ 一次问答的六步回路。

### 阶段〇：开机挂载（agent.py 启动时，只发生一次）

```
async with MCPAdapter(config) as adapter:
```

1. 按配置拉起子进程（trader 角色六座，`sys.executable servers/xxx_server.py`），
   各开一根 stdio 管道（被 MCP_TRANSPORT 点名走 HTTP 的座则是连常驻进程）
2. 每根管道先完成 MCP 握手：客户端发 `initialize` → Server 应答 →
   再发 `tools/list` → Server 把工具的**说明书**（工具名 + docstring +
   参数类型生成的 JSON Schema）递回来
3. `adapter.list_tools()` 把它们变成 LangChain 工具，喂给 `create_agent`

从此 LLM 的世界里多了这些"名片"——**它没见过函数代码，只见过说明书**
（挂几座 Server 就有几叠名片，APP_ROLE 说了算）。

### 阶段一：一次问答的六步回路

```
你> A003 还有多少？要不要补点货？
│
▼ ①组包
agent.py: history + 新问题 → {"messages": [...]} 交给 agent.astream
│
▼ ②LLM 第一醒（想）
qwen3.7-flash 收到：system prompt(供小链人设+规则) + 历史 + 问题 + 16张工具名片
输出：AIMessage(tool_calls=[{name:"inventory_query_stock", args:{sku:"A003"}}])
│    ← 这是语义路由：docstring 里写着"查询库存余量"，跟问题对上了
│      （FoodRAG 要背词表，这里 LLM 自己认路）
▼ ③过管道（做）—— MCP 协议本体，就这一下
MCPAdapter 把调用序列化成 JSON-RPC，写进子进程的 stdin：
┌─────────────────────────────────────────────────┐
│ {"jsonrpc":"2.0","id":7,"method":"tools/call",  │
│  "params":{"name":"query_stock",                │
│            "arguments":{"sku":"A003"}}}         │
└─────────────────────────────────────────────────┘
inventory_server.py 从 stdin 收到帧 → FastMCP 派发给 query_stock 函数
→ 现读 data/inventory.json（无状态，每次现翻账）
→ 找到 A003：46 个，安全线 100
→ 结果从 stdout 原路写回（JSON-RPC 响应帧）
│
▼ ④回填（看）
MCPAdapter 读到响应 → 包成 ToolMessage 塞回消息列表
   （终端上看到的两行 trace 就是这两个时刻：
    [调用工具] inventory_query_stock({'sku':'A003'})
    [账本回复] [{"sku":"A003",...,"quantity":46,"safety_stock":100,...}]）
│
▼ ⑤LLM 第二醒（再想）
LLM 看到 46 < 100：
  - 要"补货建议"就再发一轮工具调用，比如 supplier_search_suppliers("轴承")
    → 重复 ③④ → 拿到华精 7 天交期   ← 轮数由依赖决定（L3 老课）
  - 或者素材够了就直接收工
输出：不带 tool_calls 的 AIMessage = 收工轮，content 就是最终答案
│
▼ ⑥上桌
agent.py 打印答案；history.extend(本轮全部新消息) —— 下一轮"确认"才有记忆
```

实际答出来的样子（验收实录）：

> A003（深沟球轴承 6204）五金仓余量 **46 个**。
> ⚠️ 安全线 100 个，已低于安全线，建议尽快补货。

"建议补货"这半句值得盯一眼：**工具只回了数字，"46<100 该补"这个判断是
LLM 做的**——如果问题问到"找谁补、几天到"，它才会去串第二把工具。

### 三条主干结论

1. **一次工具调用 = 一根管道上一个来回**（JSON-RPC 请求帧进 stdin，
   响应帧出 stdout），N 把工具就是 N 个来回；ReAct 循环 = 想→做→看
   转几圈直到收工轮
2. **Agent 全程没碰过 JSON 文件**——数据只能从工具返回值这条门过，
   这就是甲乙方合同
3. **LLM 认工具靠 docstring，选哪把工具靠语义**——docstring 是对外 API，
   写得含糊 LLM 就走错门

## MCP 能力清单（三能力，本项目只用第一个）

| 能力 | 是什么 | 本项目 |
|---|---|---|
| **Tools** | LLM 可调用的函数 | ✅ 16 把（下表） |
| Resources | Server 暴露的只读数据文件 | ❌ 账本走工具返回，够用 |
| Prompts  | Server 预制的提示词模板 | ❌ 人设住 agent.py 的 system prompt |

## 工具清单（读写分离）

| Server | 工具 | 读/写 | 设计要点 |
|---|---|---|---|
| inventory | `query_stock(sku, warehouse="")` / `list_low_stock()` | 只读 | 查不到诚实返回"未找到"+现有SKU清单 |
| inventory | `receive_goods(order_id, confirmed=False)` | **写**（站0） | 收货入库：订单→已收货 + 库存加数量；同款确认闸门 |
| inventory | `outbound_goods(order_id, confirmed=False)` | **写**（L1） | 销售出库：单→已发货 + 库存扣减；库存不足诚实拒（绝不扣成负数）；最简版扣总数不做批次 |
| supplier | `search_suppliers(keyword)` / `get_supplier(id)` | 只读 | 名称/品类/别名三处匹配；结果按交期排好序 |
| logistics | `query_logistics(tracking_no)` / `list_in_transit()` | 只读 | 单笔给全轨迹；列表给摘要防撑爆上下文 |
| procurement | `create_order(...)` / `get_order(id)` / `cancel_order(...)` | **写** | **确认闸门**（见下） |
| customer | `search_customers(keyword)` / `get_customer(id)` | 只读（L1） | 与 supplier 对称；品类逐项子串匹配（"轴承"命中"深沟球轴承"） |
| sales | `create_sales_order(...)` / `get_sales_order(id)` / `cancel_sales_order(...)` | **写**（L1） | 与 procurement 对称：同款闸门/状态机；下单不查库存（先接单后备货） |

## 六座 MCP Server 详介

上面是总表，这里逐座展开：每座 Server 就是一位管账的师傅，工具是他的手艺，
合同写在 docstring 里。**三座带笔（inventory / procurement / sales——收货、
采购、卖出三本写账），三座只读（supplier / logistics / customer——翻账不动笔）**；
带笔的都过同款确认闸门。

### 1. inventory — 库存管家（servers/inventory_server.py）

管 `data/inventory.json`：12 条 SKU，字段含仓库、数量、单位、安全线。
你问"A003 还有多少"、"什么快用完了"，都是他的活。**站 0 起多了支笔：
货到签收入库（receive_goods）也是他的活**——货进的是他的仓库。

| 工具 | 参数 | 返回 | 合同与要点 |
|---|---|---|---|
| `query_stock` | `sku`（必填，如 "A003"）；`warehouse`（可选，限定仓库，留空查全部） | 命中：该 SKU 完整记录；未命中：`"未找到 SKU xxx"` + **现有全部 SKU 编号清单** | 编号不分大小写；查不到不编数，还把菜单递给你引导换词重试 |
| `list_low_stock` | 无 | 低于安全线的全部物料，**按缺口从大到小排序**；没有则明说 | "快用完了"的裁判是代码（`quantity < safety_stock`），不是 LLM 数感 |
| `receive_goods` | `order_id`（必填）；`confirmed`（默认 **False**） | 闸门关：拒写 + 复述提示；闸门开：`{订单号, 状态=已收货, received_at, sku, received_quantity, stock_after}` | 同款确认闸门；只收"已下单"的单（已收货→幂等提示，已取消→拒，不存在→诚实"未找到"）；**跨账本写**：同时改 orders.json 和 inventory.json |

验收实证：问"A003 还有多少"→ 答 46 个，还主动提醒低于安全线 100。
站 0 实证："PO-2026-1001 的货到了，帮我签收入库"→ 复述要素问确认 →
用户"确认" → A003 46→246，订单变已收货。

### 2. supplier — 供应商情报员（servers/supplier_server.py）

管 `data/suppliers.json`：6 家供应商，字段含品类、交期、评级、价格档、
联系方式、**别名**（学 FoodRAG 锁菜别名："华精"也能找到"华精轴承制造"）。

| 工具 | 参数 | 返回 | 合同与要点 |
|---|---|---|---|
| `search_suppliers` | `keyword`（必填） | 命中：匹配列表**按交期从短到长排好**；未命中：`"未找到"` + **现有全部品类清单** | 关键词三处匹配：名称 / 品类 / 别名；排序在代码里做——"交期最短是谁"这类问题 LLM 拿到第一行就是答案，不给它排序出错的机会 |
| `get_supplier` | `supplier_id`（必填，如 "SUP-001"） | 单家完整档案（含联系方式）；未命中明说 | 下单前核实供应商身份用它 |

验收实证："谁家供轴承？交期最短的是哪家"→ 一发命中两家，华精 7 天排榜首。

### 3. logistics — 物流追踪员（servers/logistics_server.py）

管 `data/logistics.json`：4 张运单，含轨迹事件流（时间/地点/描述）、
状态（在途/派送中/已签收）、ETA。

| 工具 | 参数 | 返回 | 合同与要点 |
|---|---|---|---|
| `query_logistics` | `tracking_no`（必填，如 "LG-2026-0001"） | 单笔全量：状态 + 起讫地 + ETA + **完整轨迹事件**；未命中：`"未找到"` + 现有运单号清单 | 快递查询的本质是事件流，按时间正序原样给 |
| `list_in_transit` | 无 | 所有未签收运单的**七字段摘要**（截掉轨迹全文），按 ETA 从早到晚排 | 单查给全文、列表给摘要——防几十条轨迹撑爆 LLM 上下文 |

验收实证："LG-2026-0001 到哪了"→ 广州分拨中心 + ETA 11-19。

### 4. procurement — 采购员（servers/procurement_server.py，唯一带笔的）

管 `data/orders.json`：采购订单账。**三件套里唯一会写文件的 Server**，
所以也是唯一设闸门的。

| 工具 | 参数 | 返回 | 合同与要点 |
|---|---|---|---|
| `create_order` | `supplier_id`、`sku`、`quantity`（必填）；`confirmed`（默认 **False**） | 闸门关：拒写 + 复述要素提示；闸门开：新订单完整记录 | **确认闸门**见下；写前先验证供应商和 SKU 真实存在（防幻觉进账本）；发号 `PO-2026-100x` 取最大号 +1 |
| `get_order` | `order_id`（必填） | 单笔订单详情；未命中明说 | 查单免费随便调 |
| `cancel_order` | `order_id`（必填）；`confirmed`（默认 **False**） | 闸门关：拒写 + 复述提示；闸门开：状态改"已取消" | 与 create 同一道闸；已取消的单再取消 → 诚实说"已经是已取消状态"，幂等不重复动账 |

**确认闸门**（本项目安全链，住工具内不住 prompt）：`confirmed=False` 时
绝不写账，返回"请先向用户复述供应商/品名/数量并取得明确同意，再带
confirmed=true 重新调用"。LLM 由工具返回值教会守规矩——prompt 是请求，
闸门是物理锁。

验收实证：下单 → 先复述要素问确认 → 用户"确认" → 才真正写账
（PO-2026-1003）；取消同款流程。

### 5/6. customer + sales — 卖手侧两座（L1 新增，与买手侧镜像对称）

买手侧是 supplier（情报）+ procurement（写账），卖手侧照样子长出
customer（情报）+ sales（写账）——**新增一个域不是改老代码，是长新 Server**。

- `customer_server.py` 管 `data/customers.json`（5 家 mock 客户，带别名）：
  `search_customers`（名称/品类/别名匹配，按编号稳定排序）、`get_customer`。
  一处不同：供应商品类存品类词（"轴承"整词可命中），客户品类存货名
  （"深沟球轴承"），必须**逐项子串匹配**——eval 测18 逮的真 bug。
- `sales_server.py` 管 `data/sales_orders.json`（SO-2026-200x 发号）：
  `create_sales_order` / `get_sales_order` / `cancel_sales_order`，
  闸门合同与 procurement 一字同款（好的合同抄过来就用）。
  设计取舍：**下单不查库存**（先接单后备货），库存够不够是出库时的事。

**验收实证（L1）**：卖给宝安五金 200 个 A004 → 复述要素问确认 → 用户"确认"
→ SO-2026-2003 入账 → 出库问确认 → A004 2600→2400；
buyer 模式问"卖 100 个轴承"→ 礼貌说明本公司不做销售（工具名片里没有销售工具）。

### 六座合起来的分工美学

- **一域一 Server 一账本**：库存问管家、供应商/客户问情报员、物流问追踪员、
  下单找采购员/销售员——LLM 按工具 docstring 自己找到对的人（语义路由）。
- **买手侧卖手侧镜像对称**：supplier:customer = procurement:sales。
  对称不只是好看——新域照样子长，老代码一行不动（L1 的全部新增都在新文件里）。
- **工具名自动带前缀**：挂载后叫 `inventory_query_stock`、`supplier_search_suppliers`
  ……多座 Server 同名工具不打架，命名空间白送。
- **写权限按域划分**：下单/取消是采购域的活，开销售单是销售域的活，
  收货入库/销售出库是库存域的活（货进出都是库存的仓库）。
  代价是跨账本写（库存管家两只手伸进采购账和销售账）：**半写/丢更新两案**
  已立案演示（见"学习路线·L0"），真药方（事务）记录在 supply_plan.md。

## 运行方法

```bash
conda activate study_ai
python agent.py                    # trader（默认）：中间商，六座全挂
APP_ROLE=buyer python agent.py    # 只买不卖（买手侧 + 核心）
APP_ROLE=seller python agent.py   # 只卖不买（卖手侧 + 核心）

# L2 混合传输：先自己把某座 Server 跑成 HTTP 常驻（.env 需 MCP_API_KEY）
python servers/inventory_server.py --http        # 8101，ctrl+c 关
MCP_TRANSPORT=inventory:http python agent.py     # inventory 走 HTTP，其余 stdio
# MCP_TRANSPORT=inventory:http,sales:http        # 逗号分隔可多座

# L3 Web：浏览器聊天页（.env 需 WEB_API_TOKEN，浏览器右上角填同款钥匙）
python web.py                                   # → http://127.0.0.1:8000

# L5 Docker：七个容器一次编排（.env 的三把钥匙自动注入）
docker compose up --build -d                    # → http://localhost:8000
docker compose ps                               # 看健康状态
docker compose down                             # 停全家（账留在卷里）
# ⚠️ docker compose down -v 会连卷清掉 = 六本账归零！平时停服务别带 -v
```

**APP_ROLE 挂载即开关**（需求 9：进货出货可选可拆）：角色只决定两件事——
挂哪几座 Server（CORE=inventory+logistics 两头共用，BUY=supplier+procurement，
SELL=customer+sales）+ system prompt 里怎么介绍公司业务边界。**Server 代码
一行不动**——"不挂载 = 不存在"，LLM 工具名片里没有销售工具，用户要卖货
它只能礼貌拒绝（buyer 模式实测）。

前置：`.env` 里有 `DASHSCOPE_API_KEY`（模板见 `.env.example`，L2 起另有
两把钥匙 `MCP_API_KEY`/`WEB_API_TOKEN`）；依赖 `langchain[mcp]`（已装，
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

## 学习路线（study_plan.md，L0~L5）

> 2026-11-17 起本项目定位为**学习作业**（JSON + mock，能用 mock 就 mock）。
> 主线：先让旧设计的毛病暴露，再上新手段治它。真实系统完整方案另存
> [supply_plan.md](supply_plan.md)。

| 站 | 主题 | 学什么 | 状态 |
|---|---|---|---|
| L0 | 业务闭环：收货入库 + eval 考卷 | 写工具闸门 / 回归考卷 / JSON 账本的坑 | ✅ 2026-11-17 |
| L1 | 卖出闭环 + APP_ROLE | 长出新域（对称设计）/ 挂载即开关 | ✅ 2026-11-17 |
| L2 | HTTP 常驻（**混合传输**） | 按座迁 stdio→HTTP / 生命周期对比 / kill 实验 | ✅ 2026-11-17 |
| L3 | FastAPI Web | 浏览器聊天页 + 服务端会话 | ✅ 2026-11-17 |
| L4 | 工程化 lite | config 集中 / 锁版本 / 日志 | ✅ 2026-11-17 |
| L5 | Docker Compose（毕业站） | 容器化部署 / 卷持久化 / healthcheck | ✅ 2026-11-17 |

### L0（原站 0）：收货入库 + eval 考卷（2026-11-17 全绿 ✅）

**业务缺口**：PO-2026-1001 在途 200 个轴承，签收了库存却不涨——缺一把
收货入库的工具。补上 `receive_goods`（住 inventory，合同与 create_order
同款确认闸门），订单状态机从"已下单⇄已取消"两态长出第三态"已收货"。

**写权限的域观念**：收货入库是库存域的活，所以工具住 inventory 而不是
procurement——写权限从"一个 Server 一支笔"变成"每域管自己的写"。
代价是**跨账本写**（inventory 伸手改 orders.json），由此故意埋下两案：

1. **半写**：receive_goods 要写两次文件（先订单账后库存账），两写之间
   崩 = 单收了货、库存没涨，两本账对不上（月底盘点才炸）。
2. **丢更新**：两个收货员并发收同一 SKU，各自"读一份→改这份→存这份"
   都没犯规，但后存的整份覆盖把先存的冲掉——期望 346 实际 246。

两案在 `scripts/lost_update_demo.py` 可复现立案（跑在临时副本上，真账不动），
**都是教学埋雷不是 bug**——站 1 换 SQLite 用事务+锁结案。

**eval.py 回归考卷**（17 题全绿，不烧 LLM）：六连测的函数级等价（查询组
①~⑨）+ 确认闸门（⑩~⑬，账本必须零变化）+ 写账组（⑭~⑰，含收货全链
46→246→346 和状态边界）。隔离手法：临时目录账本副本 + monkeypatch 各
Server 路径常量，跑完真账一字未动。惯例：每个真实 bug 进考卷当回归锚点。

验收实录：eval 17 题全绿；CLI 端到端"下单→确认→收货→确认→查库存 246"
两道闸门都灵（测完账本还原）；两案演示均可复现。

### L1：卖出闭环 + APP_ROLE 三模式（2026-11-17 全绿 ✅）

**长出一个新域**：卖手侧两座 Server（customer 情报 + sales 写账）与买手侧
镜像对称，采购/库存/物流老代码零改动（新增全在新文件）。库存管家多了
出库的手：`outbound_goods` 与 `receive_goods` 同款闸门，**库存不足诚实拒，
绝不扣成负数**；出库最简版直接扣总数，批次/先进先出留给真实系统。

**APP_ROLE 挂载即开关**（需求 9 架构验证）：trader/buyer/seller 三模式
只改挂载配置和 system prompt。buyer 实测：工具名片里没有销售工具，
"卖 100 个轴承"被礼貌拒绝。

**eval 扩到 26 题**（卖出组 ⑱~㉖：客户搜索/闸门/出库扣减/库存不足拒/
双向闭环 收→卖→出 246→196/取消幂等）。**L1 逮住两个真 bug 进考卷**：
① 采购 cancel_order 放行"已收货"单（站 0 长出第三态后漏拦，违反自家
docstring）② 客户品类匹配用整元素相等，"轴承"命中不了"深沟球轴承"
（改为逐项子串匹配）。

验收实录：eval 26 题全绿；trader 端到端"卖给宝安五金 200 个 A004 →确认→
SO-2026-2003 → 出库确认 → 2600→2400"；buyer 模式拒卖货 + 核心查询正常
（测完账本还原）。

### L2：HTTP 常驻·混合传输（2026-11-17 全绿 ✅）

**混合而不是全家切换**：六座 Server 都长了 `--http` 入口（共用
`servers/http_entry.py`：静态 Bearer 认证 + 自挂 `/health` + uvicorn），
但 agent 按座点名谁走 HTTP——`MCP_TRANSPORT=inventory:http`（逗号分隔
可多座，默认全 stdio）。mcpServers 配置表里 stdio 条目写 `command/args`、
HTTP 条目写 `url + headers`，住同一张表互不打扰：**传输是每座 Server 自己
的事，换传输只改配置不改业务代码**（协议标准化红利的活证明）。

**生命周期两种并存**：stdio Server 是 agent 的孩子（agent 退出它跟着死），
HTTP Server 是独立个体（自己起自己活，端口 8101~8106）。本站只迁
inventory（写 Server、跨账本写最重，最配常驻），其余五座留 stdio——
生产上没人一刀切全换，都是一座一座迁，本站排练的就是这个。

**kill 实验（故障隔离眼见为实）**：会话中途杀掉 HTTP 的 inventory →
查库存连续 MCPError（LLM 重试 4 次后如实告诉用户出错了）→ **同一会话里
stdio 的供应商查询照常干活** → 重启 inventory 座 → 恢复。单崩单修，
一座死不拖垮全家。

**踩坑（fastmcp 4.2 实测）**：① 无内置静态 token 认证——自己 subclass
`TokenVerifier`，`verify_token` 对上发 `AccessToken`、对不上返回 None
（框架转 401）；② `AccessToken.scopes` 是必填字段，漏了直接 500；
③ 无内置 `/health`（404）——`mcp.http_app()` 返回 Starlette app，
路由表自己 append；④ 配置校验的 transport 字面量是连字符
`streamable-http`，写下划线 `streamable_http` 直接 pydantic 报错。

验收实录：eval 26 题全绿；混合模式四连问全对（HTTP 座查库存/低库存 +
stdio 座搜供应商/查物流）；`curl /health` 200、无 key POST 401、带 key 200；
kill 实验三段全复现；纯 stdio（不设 MCP_TRANSPORT）复测不退化。

### L3：FastAPI Web 接口 + 服务端会话（2026-11-17 全绿 ✅）

**两个门脸一个后厨**：agent.py 抽出两个公共件——`build_agent(role)`
（异步上下文管理器：进 MCPAdapter → list_tools → create_agent）和
`run_turn(agent, history, question, on_event)`（单轮跑手：astream 截获
trace，返回答案+新消息+trace）。CLI 的 main() 和 web.py 的 lifespan
共用同一套；on_event 回调让 CLI 实时打印 trace、web 只收 JSON——
同一个跑手，两种门脸。

**服务端会话（与 FoodRAG 对照）**：FoodRAG 是前端持历史每次带回
（服务端零状态）；本项目反着来——会话住服务端 `data/sessions.json`
（gitignore，运行时账），前端只报 session_id，刷新/换设备不丢。
**金鱼记忆**：喂 agent 只取最近 2 轮纯文本（user/assistant 成对），
确认闸门跨一轮刚好够（Web 实测：下单复述→"确认"→落单，链路不断）；
CLI 是终端学习场景才全量喂（含工具轨迹）。

**两把钥匙**：`WEB_API_TOKEN`（外网门：浏览器→web.py）与
`MCP_API_KEY`（内网门：web→MCP Server）分开——外网钥匙泄露换外网的，
不用动内网。所有 `/api/*` 过 Bearer 校验，无 token 401。

**其余教学点**：失败轮不进账（LLM 出错的轮次不写会话，避免半截
对话污染下一轮上下文，同 CLI 纪律）；sessions.json 守读改存纪律 +
asyncio 锁串行化对话轮（学习版并发答案，真药方在 supply_plan.md）；
首问前 20 字自动当会话标题；trace 随 assistant 消息存档，回放可折叠。

验收实录：eval 26 题全绿；Web 五连问全通（查库存 46+安全线提醒 →
下单复述不写账 → "确认"落单 PO-2026-1003 → 取消复述 → "确认"取消）；
双会话隔离（会话 2 对会话 1 的对话一无所知）；sessions.json 落盘可查
（2 会话/标题/trace 齐全）；CLI 复测不退化（16 把工具照挂）。

### L4：工程化 lite（2026-11-17 全绿 ✅）

**config.py 配置集中**：所有"从环境读东西"只住一个文件——路径
（DATA_DIR/LOG_DIR/SESSIONS_FILE）、角色（APP_ROLE）、传输开关
（MCP_TRANSPORT）都是常量，密钥走 `require_secret()`（缺钥匙
fail fast，把缺的一次报全，别一把一把挤牙膏）。之前 load_dotenv
散在 agent/web/http_entry 三处各写一遍——改一处忘两处的温床。

**两份工程文件**：`.env.example` 进库当模板（真 .env 继续 gitignore，
新人 cp 一份填钥匙就能起）；`requirements.txt` 实测版本钉死
（langchain 1.4.6 / fastmcp 4.2.0 / fastapi 0.121.3 / uvicorn 0.38.0，
**starlette<0.51 红线**写进注释——fastmcp 想抬到 1.x 会弄坏 fastapi）。

**JSON 行日志**：`logs/web.log`（web 一份）+ `logs/mcp_<名>.log`
（每座 HTTP Server 一份）。web 侧经 L3 埋的 `on_event` 回调长成
日志管线：`chat_start → tool_call → tool_result（带 duration_ms）→
chat_end`，每行带 session_id——`grep <session_id> logs/web.log`
一条命令串起一次对话的全链。MCP 侧 uvicorn 访问日志同款 JSON
格式（谁来了/什么状态/建了哪个会话）。给机器读的账（对账/排查）
和给人看的 CLI print trace 两不耽误。

验收实录：eval 26 题全绿；`python web.py` 一键起（日志目录自动建）；
grep session_id 串全链（chat_start/tool_call 38ms/chat_end 6581ms）；
MCP HTTP 座日志落盘（health 200/带 key 200/无 key 401 全记录）；
CLI 复测不退化；fail fast 冒烟（缺钥匙当场退出并指路 .env.example）。

### L5：Docker Compose 毕业站（2026-11-17 全绿 ✅）

**一个镜像七场演出**：`Dockerfile`（python:3.11-slim + 锁版本依赖 +
拷代码和种子账本）构建一次，compose 用不同 `command` 起七个容器——
六座 MCP Server 全 `--http`（容器内听 0.0.0.0）+ web（唯一对宿主机
开 8000 端口，六座 MCP 只在容器内网互访）。**密钥不进镜像**：
.env 在 .dockerignore 挡掉，三把钥匙走 compose environment 注入
（镜像会被分发，钥匙跟着镜像走 = 见者有份）。

**容器网络 DNS（又一次"只改配置不改代码"）**：web 容器里 agent 连
MCP Server 用的是服务名 `http://inventory:8101/mcp`——compose 自带
DNS，服务名即主机名。agent.py 只认 `MCP_HTTP_HOST=service` 这个
环境变量约定（url 主机名换成 Server 名），业务代码零改动。

**卷持久化**：`supply-data`（六本 JSON 账 + 会话账）和 `logs` 两个
命名卷挂进所有容器；首次起卷时 Docker 把镜像里的种子 data/ 拷进
空卷，之后容器随便杀随便重建，账本活在卷里。
**healthcheck 链**：每座 MCP 打 `/health`，web `depends_on:
service_healthy`——六座全部探活通过 web 才起，不让它在 Server
没醒时就去连。

**依赖三步装法（本站最大的坑）**：容器里干净解析天生不可能——
fastapi 0.121.3 钉死 starlette<0.51，fastmcp 4.2（经 fastmcp-slim
[client,server]）要 >=1.0.1，ResolutionImpossible。宿主机当初是
"先装齐再降级"救回来的共存态；Dockerfile 照方抓药：①摘掉
fastapi/starlette 两行装齐全家 ②强制压回 starlette 0.50.0
③最后装 fastapi。pip 中途警告依赖冲突——警告不是错误，运行时
真不兼容才炸，而这个组合 L2~L4 已全部实测。

> ⚠️ **红字警告：`docker compose down -v` 会连卷一起清掉——六本账
> 全部归零！** 平时停服务用 `down`（不带 -v）或 `stop`。验收时
> 亲手试过一次：down -v 后 up，账本回到镜像种子状态。

验收实录：`docker compose up --build -d` 一键起全家（六座 healthy
后 web 才启动）；localhost:8000 **六连测全通**——查库存/低库存/
供应商比价/物流轨迹全准，下单复述→"确认"→PO-2026-1003 落单→
取消复述→"确认"→已取消（确认闸门在"web 容器→HTTP→MCP 容器"
全链上照样灵）；**kill 实验**：`compose kill` 全家被杀 → `up -d`
→ 卷里订单、16 条会话消息分毫无损，接着聊 LLM 还记得那单已取消；
宿主机真账全程零接触（写脏的只是卷——容器化天然隔离的红利）；
`down -v` + `up` 验证卷重置（种子重新拷入）。

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

## MCP 落地全流程：设计 → 开发 → 接入 → 调试 → 测试 → 上线

把本项目亲手趟过的路整理成方法论。每个阶段两栏：**本项目怎么做的**（活例子）
和 **上生产要换的脑子**（Mock 与真实的差距在哪）。

| 阶段 | 一句话 | 本项目活例子 | 上生产升级点 |
|---|---|---|---|
| ① 设计 | 甲方想清楚卖什么能力 | 6 本账 → 6 个 Server，16 把工具读写分离 | 能力清单评审：谁可调用、限流多少 |
| ② 开发 | Server 侧把工具写好 | FastMCP `@tool()`，docstring 即合同 | 真数据库、并发锁、事务 |
| ③ 接入 | 客户端挂载 + 教 LLM 规矩 | MCPAdapter config + system prompt | 远程 HTTP 接入 + 认证 |
| ④ 调试 | 分层定位：Server 单练还是合练出问题 | trace 两行打印 + 哈希比对 | Inspector + 日志链路追踪 |
| ⑤ 测试 | 函数级 → 端到端 → 彩蛋 | 闸门单测 + 六连测 + 语义路由 | 回归考卷 + CI 每次必跑 |
| ⑥ 上线 | 本地子进程 → 常驻服务 | L2 混合传输（stdio+HTTP 按座选）→ L5 Docker 七容器 | 云上部署 + 监控 + 版本化 |

### ① 设计：甲方想清楚卖什么能力

**先定能力，再写代码。** 三个决策按顺序做：

1. **能力选型**：MCP 三能力（Tools/Resources/Prompts）里卖哪个？
   本项目只卖 Tools——LLM 要的是"帮我查/帮我办"，不是"给我看文件"。
2. **切 Server**：按业务域切，一域一 Server 一账本。库存/供应商/物流/采购
   /客户/销售六个域天然独立，单崩单修，将来单独替换任何一个都不惊动别人
   （L1 卖出闭环就是活证据：长出两座新 Server，旧四座一行没动）。
3. **每把工具的合同**，五问定死：
   - 读还是写？（读随便调；写要过闸门——读写分离是第一道安全线）
   - 参数是什么类型？（`sku: str`、`quantity: int`——类型即合同，LLM 照 schema 填空）
   - 查不到返回什么？（诚实说"未找到"，绝不编——防幻觉从工具层做起）
   - 写之前要什么手续？（确认闸门：`confirmed=False` 拒写并说明补救路径）
   - 返回多少？（列表给摘要防撑爆上下文，详情走单查——logistics 的设计）

**上生产的脑子**：加两问——这工具谁有权调？（认证授权）调疯了怎么办？（限流、
审计日志）。写工具的闸门从"复述确认"可能要升级为"审批流"（下单超 1 万元
走人工审批）。

### ② 开发：Server 侧把工具写好

```python
mcp = FastMCP("inventory")          # 一个 Server 一个身份

@mcp.tool()
def query_stock(sku: str, ...) -> str:
    """docstring 就是给 LLM 看的说明书"""  # 什么时候调、参数含义、合同条款
    ...                                # 函数体：读账 → 加工 → 返回
```

铁律四条（全部本项目实证）：
- **docstring 即合同**：LLM 只看函数签名 + docstring 决定怎么用你。写不清
  等于合同有歧义，甲乙方迟早扯皮（确认闸门的条款就写在 docstring 里）。
- **stdout 是协议通道**（stdio 传输下），调试信息只能走 stderr。
- **写账纪律**：读一份 → 改这份 → 存这份；写前先验证外键真实存在
  （供应商/SKU 不存在就拒单，防幻觉进账本）。
- **无状态**：每次现读账本，Server 随便重启不失忆——代价是磁盘 IO，
  换鲁棒，值。

**上生产的脑子**：JSON 账本换成真数据库（并发写靠锁/事务，两个订单同时
发号不会撞号）；"无状态现读"在大流量下升级为"缓存 + 失效策略"。

### ③ 接入：客户端挂载 + 教 LLM 规矩

两件事，一件对协议，一件对模型：

```python
config = {"mcpServers": {
    "inventory": {"command": sys.executable,   # 防环境飘移，见教学沉淀 4
                  "args": [".../inventory_server.py"],
                  "transport": "stdio"}}}
async with MCPAdapter(config) as adapter:
    tools = await adapter.list_tools()          # 16 把工具到手
    agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)
```

- **协议侧**：MCPConfig 描述怎么拉起/连上每个 Server；`list_tools()` 拿到的
  工具名自动带前缀（`inventory_query_stock`）——多 Server 同名工具不打架，
  命名空间白送。
- **模型侧**：system prompt 教规矩（人设 + 判单规则 + "下单必须先取得同意"）。
  工具 docstring 管"这把工具怎么用"，system prompt 管"整个公司怎么运转"，
  两层各管各的。
- **历史自己记**：create_agent 不记上一轮，对话历史手动累积每轮全量喂
  （踩坑实录第 3 条）。

**上生产的脑子**：本地 stdio 换远程 HTTP——config 从 `command/args` 换成
`{"url": "https://...", "transport": "streamable-http"}`，Server 常驻运行，
再加认证头。接入代码只改 config 不改逻辑，这就是协议标准化的红利。
**L2 已亲手排练**：`MCP_TRANSPORT=inventory:http` 按座点名，混合模式下
六连测全绿；L5 容器里六座全走 HTTP（url 主机名换成 compose 服务名，
又是零代码改动）。

### ④ 调试：分层定位

MCP 项目出问题，先分清是哪一层的锅，别一锅炖：

| 层 | 症状 | 手段 |
|---|---|---|
| Server 本身 | 工具逻辑错/账读错 | 脱离 agent 单测：直接 import 调函数（闸门单测就这么跑的） |
| 协议层 | 子进程秒死/工具列表空 | 单独 `python servers/xxx_server.py` 看报错；查 sys.executable；官方 Inspector 可视化手动调工具 |
| 接入层 | 工具在但 LLM 不调/调错 | 看 agent.py 的 trace（调用工具/账本回复两行），对照 system prompt |
| 依赖层 | 装完包别的项目坏了 | starlette 案：装新包前后 `pip check`，版本冲突当场发现 |

trace 是主力：每轮打印 `[调用工具] 名(参数)` 和 `[账本回复]`，LLM 选错了
工具、参数填错了、Server 返回了拒绝，一行看穿。

**上生产的脑子**：请求 ID 链路追踪（一次对话穿过多层，日志要能串起来）；
Server 进程崩溃自动重启 + 告警。

### ⑤ 测试：三层金字塔

1. **函数级单测**（快、免费）：直接调工具函数验合同——`confirmed=False`
   必须拒写、查不存在的 SKU 必须说未找到。不烧 LLM。
2. **端到端验收**（慢、烧 token，但必须跑）：真实对话走全链路，六连测就是
   这层——重点测**行为**而不是字句（"必须先复述订单要素问确认"）。
3. **彩蛋/边界测试**：语义路由（无关键词措辞）、恶意/误操作（对已取消的
   单再取消一次）。

**惯例**（学 FoodRAG eval.py）：每个真实 bug 进考卷当回归锚点，改完代码
必跑全卷。

**上生产的脑子**：考卷进 CI，每次改 Server 自动跑函数级，端到端抽样跑。

### ⑥ 生产上线：本地子进程 → 常驻服务

本项目形态（L2~L5 演化后）：**默认本地 stdio，按座可切 HTTP 常驻，L5 起
还有 Docker 七容器整 deploy**——同一份代码三种跑法。安全靠三道：确认闸门
（协议内的锁）+ Bearer 静态 key（HTTP 座的门禁，L2）+ 容器网络隔离
（六座 MCP 只在内网互访，唯一对外只开 web 的 8000，L5）。

本项目已经排练过的换血（Mock → 生产对照，左列全部亲手做过）：

| 本项目已排练 | 生产版还要加 |
|---|---|
| stdio 子进程 + HTTP 常驻按座并存（L2 混合传输） | 云上部署、按流量独立扩容 |
| Bearer 静态 API key（MCP 内网一把 + Web 外网一把） | OAuth / 按角色限工具 |
| JSON 文件账本 + 卷持久化（Docker 重启不丢账） | 数据库 + 事务 + 备份（另立仓库的真系统上 SQLite） |
| 写前复述确认 | 审批流 + 审计日志（谁在何时下了什么单） |
| JSON 行格式日志，session_id 可 grep 串链（L4） | 监控大盘 + 告警 |
| Docker Compose healthcheck + 自动重启（L5） | 版本化：schema 变更 = 合同变更，老客户端要兼容期 |

最后一行最容易踩：工具的参数/返回一改，所有挂载它的 Agent 端同时受影响——
**Server 的 docstring 不是注释，是对外 API**。

## 谱系对照（这个项目站在哪些前人的肩膀上）

| 设计 | 学的是 |
|---|---|
| JSON 账本 + 发号（PO-2026-1003+1） | FoodRAG 账本风格 / backfill_ids 发号思想 |
| 供应商别名匹配（"华精"→华精轴承制造） | FoodRAG 锁菜别名 |
| 查不到诚实返回"未找到" | FoodRAG 防幻觉 / drug_rag 诚实拒答 |
| 确认闸门住工具返回值 | drug_rag 安全链"约束住离危险最近的地方" |
| LLM 语义路由 vs 词表路由 | FoodRAG router 的反面教材，Agent 的长处 |
| system prompt 人设+规则表 | 陈玉楼算命 agent |

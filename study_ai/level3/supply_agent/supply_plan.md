# supply_agent 实现计划（需求升级版）

> 配套文档：[requirements.md](requirements.md)（需求定稿与架构决定）。
> 本计划 = 双轨合一：业务加宽（站 1~5）+ 生产化（站 6~10，原六站路线的
> 站 2~6 平移）。**每站独立提交 main，验收绿了才提交；不 push 除非用户说。**

## 总路线图

| 站 | 主题 | 一句话 | 状态 |
|---|---|---|---|
| 0 | 业务闭环（收货入库 + eval 考卷） | ✅ 已完成（b4aea12） | ✅ |
| **1** | **地基：SQLite + 全域 schema** | 换账本材质，追溯链的键一次埋齐 | ⬜ 下一步 |
| 2 | 卖出闭环 | 客户 + 销售单 + 出库，中间商两头通 | ⬜ |
| 3 | 买手数据 | 报价 + 批次进价 + 质检 | ⬜ |
| 4 | 合同域 | 条款存储 + LLM 审查 + 全链追溯 | ⬜ |
| 5 | 报表域 | 只读聚合 Server（考评/统计/利润） | ⬜ |
| 6 | HTTP 常驻 | stdio → HTTP + API Key 认证 | ⬜ |
| 7 | FastAPI Web | 浏览器聊天页 + 服务端会话 | ⬜ |
| 8 | 工程化 | config 集中 / JSON 日志 / 锁版本 | ⬜ |
| 9 | Docker Compose | 一键起全家 + 卷持久化 | ⬜ |
| 10 | 云部署 | 前置清单（不含施工） | ⬜ |

---

## 站 1：地基——JSON → SQLite + 全域 schema

**目标**：站 0 两案（半写/丢更新）用事务结案；8 条需求的关联键一次埋进库。

**改动**：
1. `db.py`：连 `data/supply.db`（环境变量 `SUPPLY_DB_PATH` 可覆盖——eval/测试库
   的钥匙）；每连接 `PRAGMA journal_mode=WAL` + `busy_timeout=5000`；
   `receive_goods_txn()` 单事务收货（`BEGIN IMMEDIATE` → 改订单 → 加库存 →
   写 audit → `COMMIT`）
2. `scripts/seed.py`：JSON → SQLite 搬运工（`INSERT OR REPLACE` 幂等），
   JSON 文件保留当种子格式
3. **全域 schema**（新建表骨架 + 现有表埋键；施工前 DDL 先给用户过目）：

   | 表 | 用途 | 关联键（本站必埋） |
   |---|---|---|
   | suppliers / customers | 两头交易对象 | — / 新建骨架 |
   | contracts | 合同（采购/销售） | party_id；新建骨架 |
   | purchase_orders | 采购单 | **contract_id NULL**、expected_at NULL |
   | sales_orders | 销售单 | **contract_id NULL**、customer_id |
   | inventory | 现量余额 | 不动（余额仍由流水推导校验） |
   | lots | 入库批次 | order_ref → 采购单；unit_cost、quantity_remaining |
   | movements | 出入库流水 | lot_id、direction、order_ref |
   | quotations | 报价 | supplier_id + sku + 有效期 |
   | inspections | 质检 | lot_id |
   | audit_log | 审计 | 三个写工具每次动笔记一行 |

4. 四个 Server 换读写函数：`_load/_read/_write` → `db.load_table` 等具体函数；
   **工具合同（docstring/参数/文案）一字不动**——伏笔兑现的活证据
5. `eval.py` 适配：隔离钥匙换 `SUPPLY_DB_PATH` 指临时库 + seed，
   **原 17 题断言一字不改**；加题 ⑱并发收货对打（A003 恰好 346）
   ⑲审计行数对账（N 次写操作 = audit_log N 行）
6. `scripts/crash_test.py`：子进程事务内 commit 前 `os._exit(1)` →
   父进程查库两表都没变（半写案终审）

**做到什么程度**：新表只建骨架不建工具（customers/contracts 等先空着）；
现有 10 把工具行为零变化。

**验收标准**：
1. `python scripts/seed.py && python eval.py` 19 题全绿
2. `python scripts/crash_test.py` 无半写（两表都没变）
3. CLI 手测六连测复测不退化（对话行为零变化）
4. schema 过目：DDL 给用户 review 后才施工
5. 提交：`站1：JSON→SQLite+全域schema，事务治两案，audit_log起步`

---

## 站 2：卖出闭环——客户 + 销售单 + 出库

**目标**：中间商两头通。卖出域与采购域完全对称（同款确认闸门、同款状态机）。

**改动**：
1. `data/customers.json` 种子（mock 客户若干家，带别名——学供应商账）
2. 新 Server `sales_server.py`：
   `create_sales_order(customer_id, sku, quantity, confirmed)` /
   `get_sales_order` / `cancel_sales_order`（闸门合同同款）
3. `inventory_server` 加 `outbound_goods(order_id, confirmed)`：按**先进先出**
   扣批次（先收的批次先出，可跨批次拆）、写 movements 流水、库存余额同步扣
4. 状态机：已下单 → 已发货（出库）/ 已取消
5. **APP_ROLE 角色开关**（需求 9）：环境变量 `APP_ROLE` =
   `trader`（默认，全挂）/ `buyer`（只挂买手侧 supplier+procurement）/
   `seller`（只挂卖手侧 customer+sales）——agent.py 按角色过滤 MCP_CONFIG +
   调整 system prompt；**seller 模式配期初批次**（开仓一批货带成本，
   没有进货也能出库）
6. eval 扩题：卖出闸门 / FIFO 出库顺序（旧批次先扣）/ 跨批次拆单 /
   库存不足拒出库 / 双向闭环（收货→出库→余额对账）

**做到什么程度**：卖出的货**能出能追**（出库记录到批次），先不碰利润计算（站 5）；
三种角色模式可切换可运行。

**验收标准**：
1. eval 全绿（含新增卖出组）
2. CLI 端到端：卖给客户 → 复述确认 → 出库 → 库存降、批次余量对账
3. **三模式各起一次**：trader 全能；buyer 问卖货被礼貌拒（工具名片里
   根本没有销售工具）；seller 靠期初批次可出库
4. 采购侧六连测复测不退化
5. 提交：`站2：卖出闭环+APP_ROLE三模式，客户+销售单+FIFO出库`

---

## 站 3：买手数据——报价 + 批次进价 + 质检

**目标**：买手侧三样关键数据开始积累（比价、质量、成本原料）。

**改动**：
1. `quotations` 表 + 种子数据；supplier_server 加 `compare_prices(sku)`：
   各供应商报价按价格排序（确定性排序在代码里做——既定惯例）
2. `receive_goods` 升级：收货即建批次（lot 发号 LOT-xxx）+ 记 unit_cost
   （从报价表带出或参数传入）
3. `record_inspection(lot_id, score, issue)`：质检记录挂批次
   （写工具，轻量确认闸门或免闸门——施工时定）
4. eval 扩题：比价排序 / 收货建批次 / 批次带进价 / 质检挂批次可反查供应商

**做到什么程度**：数据开始"跑业务时自然积累"；分析报表仍不碰（站 5）。

**验收标准**：
1. eval 全绿（含新增批次/报价组）
2. 收货全链复测：收货 → 批次生成 → 批次可反查订单和供应商
3. CLI 手测比价一发（"A003 找谁买最便宜"）
4. 提交：`站3：报价+批次进价+质检，买手数据起步`

---

## 站 4：合同域——条款存储 + LLM 审查 + 全链追溯

**目标**：需求 7/8 落地——合同进系统，货能反查合同。

**改动**：
1. `contracts` 表（结构化条款：编号/类型/对方/金额/付款周期/质保/违约条款）
   + 原文存 `data/contracts/` 目录（表里存 file_path——双层存储已定）
2. 新 Server `contract_server.py`：
   - `get_contract(id)` / `list_contracts(keyword)`（读）
   - `check_contract(file_path 或 id)`：**LLM 条款审查**——读原文，对照检查
     清单逐条给意见（检查清单本站与用户共同定稿）。LLM 调用住 Server 内
     还是住 Agent 侧，施工时验证定夺（教学点：确定性工具 vs LLM 工具的分界）
3. 订单挂合同：create_order / create_sales_order 加可选 `contract_id`
   （**工具合同变更，单独提示用户**）
4. 追溯工具 `trace_lot(lot_id)`：批次 → 订单 → 合同 → 对方，一条链拉通
5. eval 扩题：合同存取 / 追溯链完整性；LLM 审查不进 eval（非确定性），
   手测验收

**做到什么程度**：一份 mock 合同能存、能审、能从批次追到它。

**验收标准**：
1. eval 全绿（确定性部分）
2. 手测：给一份 mock 合同文本 → check_contract 出结构化意见
3. 手测追溯：问"LOT-xxx 这批货出问题了，是哪个合同的"→ 一条链答到底
4. 提交：`站4：合同域，双层存储+LLM审查+批次反查合同`

---

## 站 5：报表域——只读聚合 Server

**目标**：需求 1~6 的"看"全部落地——账本里攒的数据变成答案。

**改动**：新 Server `report_server.py`（只读，与三座只读 Server 同款）：
1. `supplier_scorecard(supplier_id)`：出货量 / 及时率（expected_at vs
   received_at）/ 平均质检分——需求 2/3/4
2. `customer_stats(period)`：月/季/年卖给谁最多、各客户量与额——需求 6
3. `profit_by_customer(period)`：卖价 − FIFO 批次成本，按客户算利润——需求 5
4. 聚合全在 SQL 里做，LLM 只负责把数字讲成人话（确定性信息代码算——既定惯例）
5. eval 扩题：用种子数据手工算好的期望值对账（数字必须分毫不差）

**做到什么程度**：报表能答需求清单里每一个"我想看见"。

**验收标准**：
1. eval 全绿（报表数字与手工对账一致）
2. CLI 手测三连：供应商考评 / 客户统计 / 利润排名
3. 提交：`站5：报表域，供应商考评+客户统计+利润`

---

## 站 6~10：生产化（原六站路线平移，需求侧不再改动）

| 站 | 内容 | 验收标准（要点） |
|---|---|---|
| 6 HTTP 常驻 | 四 Server 加 `--http`（8101~8104）+ Bearer 静态 key；agent.py 按 `APP_ENV` 切 config | 四服务活住 + /health 200 + 无 key 401 + prod 六连测全通 + dev 不退化 |
| 7 FastAPI Web | `web.py` POST /api/chat + 服务端会话（sessions.db）+ 单文件聊天页；抽 `build_agent()` 公共件 | 双会话隔离 + Web 六连测 + 历史落库可查 + CLI 不退化 |
| 8 工程化 | `config.py` + `.env.example` + JSON 行日志进 logs/ + requirements 锁版本 | 一键起 + 同 session_id 全链可 grep + eval 全绿 |
| 9 Docker Compose | 共享基础镜像；4×MCP + web 五服务；卷持久化 + healthcheck | `up --build` 一键全家 + kill 重启账本会话无损 |
| 10 云部署 | README 前置清单：服务器/域名 HTTPS/密钥管理/安全组只开 443 | 站 9 验收后另立计划 |

---

## 纪律（全计划通用）

- **每站三件套**：eval 全绿 + 本站验收项 + 上一站关键场景复测
- 工具合同（docstring/参数）尽量不动——动就是对外 API 变更，单独提示用户
- starlette 红线：任何 pip install 后 `pip check` + fastapi 冒烟
- 每个真实 bug 进 eval 考卷当回归锚点
- 端到端对话（烧 LLM）只在有新业务闭环的站手跑（2/4/5/6/7/9）
- 测试写脏真账时：先备份 data/，测完还原（站 0 已验证的做法）

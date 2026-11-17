# supply_agent 学习版计划（本仓库）

> 定位：**作业 / 教具**。目标不是做真实系统，是亲手摸懂——
> ① Agent 与 MCP 的交互全链路 ② 多 Server 整体架构（怎么长出新域）
> ③ Web 接口 ④ Docker 部署。
>
> 原则：**能用 mock 就 mock，账本用 JSON 不上 SQLite**。站 0 立的两案
> （跨账本半写 / 并发丢更新）不治——`scripts/lost_update_demo.py`
> 演示一遍达成教学目的即可，真药方（事务）记录在案但不施工。
>
> 配套：需求背景见 [requirements.md](requirements.md)；
> 真实系统完整方案见 [supply_plan.md](supply_plan.md)（另立仓库时参考）。

## 学习站路线

| 站 | 主题 | 学什么 | 状态 |
|---|---|---|---|
| L0 | 收货闭环 + eval 考卷 | 写工具的确认闸门 / 回归考卷 / JSON 账本的坑 | ✅ 已完成（b4aea12） |
| L1 | 卖出闭环（可选，推荐） | 长出一个新域：对称设计 / 挂载即开关 | ⬜ |
| L2 | HTTP 常驻（混合传输） | stdio → HTTP 按座迁移：传输是每座 Server 自己的事 | ✅ 2026-11-17 |
| L3 | FastAPI Web | 接口层：浏览器聊天页 + 服务端会话 | ✅ 2026-11-17 |
| L4 | 工程化 lite | config 集中 / 依赖锁版本 / 日志 | ✅ 2026-11-17 |
| L5 | Docker Compose | 容器化部署：一键起全家 + 卷持久化 | ⬜ |

---

## L1：卖出闭环（可选站，推荐做）

**学什么**：在活架构上长出一个新域——销售 Server 与采购 Server 完全对称
（同款确认闸门、同款状态机）；再体验"挂载即开关"（需求 9 的架构验证）。

**做什么**：
1. `data/customers.json` 种子（mock 客户，带别名——学供应商账）
2. 新 Server `sales_server.py`：`create_sales_order(customer_id, sku,
   quantity, confirmed)` / `get_sales_order` / `cancel_sales_order`
3. `inventory_server` 加 `outbound_goods(order_id, confirmed)`：**最简版**——
   直接扣 inventory 总数（不做批次/先进先出，那是真实系统的事），
   库存不足诚实拒出库
4. `APP_ROLE` 角色开关：`trader`（默认全挂）/ `buyer`（只挂买手侧）/
   `seller`（只挂卖手侧）——agent.py 按角色过滤 MCP_CONFIG + 调整
   system prompt，**只改配置不改代码**
5. eval 扩题：卖出闸门 / 出库扣减 / 库存不足拒 / 采购侧复测

**做到什么程度**：mock 到位、能跑通对话即可，不追求账目严谨。

**验收**：eval 全绿；CLI 端到端卖给客户→确认→出库→库存降；
buyer 模式问卖货被礼貌拒（工具名片里没有销售工具）；
trader 模式六连测不退化。

---

## L2：HTTP 常驻（混合传输）

**学什么**：stdio 是"Agent 拉子进程"，HTTP 是"Server 常驻、谁都能连"。
本站**不做全家切换，做混合**——一部分 stdio 一部分 HTTP，证明传输是
**每座 Server 自己的事**：mcpServers 配置里每个条目独立，这条写
`command/args`，那条写 `url + headers`，agent 根本感觉不出差别。

顺带亲眼见三个东西：
1. **进程生命周期对比**：stdio Server 是 agent 的孩子（agent 退出它跟着死），
   HTTP Server 是独立个体（自己起自己活）——一个系统里两种并存
2. **渐进迁移**：生产上没人一刀切全换，都是一座一座迁——本站就排练这个
3. **故障隔离**：kill 掉 HTTP 那座，stdio 的工具照常干活——"单崩单修"眼见为实

**做什么**：
1. 四个 Server 加 `--http` 入口：`mcp.run(transport="http", host, port,
   path="/mcp")`；端口分工 8101~8104；stdio 原样保留（默认）
2. Bearer 静态 API Key 认证（fastmcp 内置 auth 优先，不行就中间件），
   key 进 .env；`/health` 健康检查端点
3. agent.py 的 MCP_CONFIG 改成**按座指定传输**：环境变量
   `MCP_TRANSPORT=inventory:http`（逗号分隔可多座，默认 stdio）——
   被点名的座走 `url + 认证头`，其余照旧 `command/args`
4. **先迁 inventory**（写 Server、跨账本写最重，最配常驻），
   其余三座留 stdio；想再迁，config 加一行的事

**做到什么程度**：混合模式长期并存，随时按座切换。

**验收**：
1. 混合模式（inventory=HTTP + 其余 stdio）六连测全通（烧 LLM 认了）
2. HTTP 座活住：`curl /health` 200；无 key 请求 401
3. **kill 实验**：杀掉 HTTP 的 inventory → 查库存报错、查供应商/物流/
   下单照常 → 重启 inventory → 恢复
4. 纯 stdio 模式（不设 MCP_TRANSPORT）复测不退化

---

## L3：FastAPI Web 接口

**学什么**：CLI 聊天升级成浏览器聊天页；服务端会话管理（对照 FoodRAG
前端持历史的另一种做法）；agent 构建抽公共件。

**做什么**：
1. `agent.py` 抽 `async def build_agent()`（CLI 和 web 共用）
2. `web.py`：`POST /api/chat {session_id, message} → {answer, trace[]}`；
   Bearer token 认证（与 MCP key 分开——外网/内网两把钥匙）
3. 会话存 **JSON**（`data/sessions.json`，不上 SQLite）：最近 2 轮历史
   喂 agent（金鱼记忆，够确认闸门跨一轮用）
4. `static/index.html` 单文件聊天页（FoodRAG 同款套路）：会话切换 /
   trace 折叠 / 正在想提示

**做到什么程度**：双会话隔离可验证；trace 上屏。

**验收**：浏览器双会话各聊各的互不串；Web 上六连测全通（含下单确认
闸门）；sessions.json 落盘可查；CLI 复测不退化。

---

## L4：工程化 lite

**学什么**：小项目也要有的工程习惯——配置集中、依赖锁死、日志留痕。

**做什么**：
1. `config.py` 统一读 .env → 常量（MCP_TRANSPORT/APP_ROLE/端口/key/路径），
   缺关键变量 fail fast
2. `.env.example` 进库当模板（真 .env 继续 gitignore）
3. `requirements.txt` 实测版本钉死 + 教训注释（**starlette<0.51 红线**）
4. web/HTTP 模式日志进 `logs/`（每行带 ts/session_id/tool/duration_ms）；
   CLI 保留 print trace（调试口不丢）

**做到什么程度**：一键起、配置自解释、日志能 grep 串链。

**验收**：`python web.py` 一键起；`grep <session_id> logs/*.log` 能串起
一次对话全链；eval 全绿 + Web 六连测复测。

---

## L5：Docker Compose 部署（毕业站）

**学什么**：把"我电脑上能跑"变成"任何机器能跑"——镜像、容器、网络、
卷持久化、健康检查，一次配齐。

**做什么**：
1. `Dockerfile` 共享基础镜像：python:3.11-slim + requirements + 拷代码；
   **不 COPY .env**（密钥走注入）
2. `docker-compose.yml` 五个服务：4×MCP Server（`--http`，容器内
   0.0.0.0:810x）+ web（`MCP_TRANSPORT` 四座全点名 http，MCP url 指向
   **服务名**——容器网络 DNS，又一次"只改环境变量不改代码"）
3. 卷：`supply-data:/app/data`（JSON 账本持久化）+ `logs:/app/logs`；
   healthcheck 打 /health；restart: unless-stopped
4. README 红字：`docker compose down -v` 清卷毁账

**做到什么程度**：本机演练级（单机 Docker），不碰云。

**验收**：`docker compose up --build` 一键起全家 → localhost:8000
六连测全通；`docker compose kill` 后 `up -d` 账本会话无损（卷持久化实证）。

---

## 纪律（不变）

- 每站独立提交 main，验收绿了才提交；不 push 除非用户说
- 工具合同（docstring/参数）尽量不动——动就单独提示
- starlette 红线：pip install 后 `pip check` + fastapi 冒烟
- 每个真实 bug 进 eval 考卷；端到端对话在有新闭环的站手跑
- 测脏真账：先备份 data/，测完还原

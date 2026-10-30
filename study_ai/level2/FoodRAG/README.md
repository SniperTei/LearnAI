# FoodRAG — 金燕菜单迷你 RAG

药品说明书 RAG（drug_rag2.0）的同构缩微版：61 道家常菜 + 6 家饭店（一分店一行），
从读数据到答"锅包肉怎么做 / 罗湖万象城有什么工作餐"，全链路本地运行
（Ollama bge-m3 嵌入 + qwen3:4b 生成），零 API 成本。

## 目录结构

```
FoodRAG/
├── data/                          # 数据区
│   ├── jinyan_menu.json           # 菜品原料（61 道菜 × 11 字段）
│   ├── jinyan_menu.md             # 菜单人读版，程序不读
│   ├── restaurant.md              # 饭店原料（人填这张表）
│   ├── restaurant.json            # converter 产物（程序读这份）
│   ├── menu_chunks.jsonl          # 菜品账本（一行=一块）
│   ├── restaurant_chunks.jsonl    # 饭店账本
│   └── chroma/                    # 双仓库持久化（gitignore）
├── prep/                          # 备菜线（数据更新时才跑）
│   ├── convert_restaurant.py      # md→json + 质检台对账
│   ├── build_menu_chunks.py       # 菜品读+切
│   ├── build_restaurant_chunks.py # 饭店读+切
│   └── build_index.py             # 双仓量化+存
├── search.py                      # 点菜区：锁菜检索 + trace
├── ask.py                         # 点菜区：受控生成问答
├── requirements.txt
└── README.md
```

## 架构

### 备菜线（入库，数据更新时跑）

```
菜品线: jinyan_menu.json → build_menu_chunks.py → menu_chunks.jsonl ─┐
                                                                      ├→ build_index.py → 双 Chroma 仓
饭店线: restaurant.md → convert_restaurant.py(质检) → restaurant.json  │
        → build_restaurant_chunks.py → restaurant_chunks.jsonl ───────┘
```

### 点菜区（每次提问都走）

```
问题 → search.py   查：锁菜（菜名 in 问题 → filter）+ 向量检索 + trace 打印
     → ask.py      拼：证据塞提示词 → qwen3:4b 受控生成（防幻觉）
```

### 六步口诀 → 文件对照

| 口诀 | 文件 | 干的活 |
|---|---|---|
| 读 | `prep/convert_restaurant.py` + 两个切块脚本 | md/json 读入（md 先转 json 过质检台） |
| 切 | `prep/build_menu_chunks.py` / `prep/build_restaurant_chunks.py` | 一道菜一块 / 一分店一块，贴标签落账本 |
| 量化 | `prep/build_index.py` | 无独立代码，外包给 Chroma 的 add_texts |
| 存 | `prep/build_index.py` | 双仓持久化，整仓删除重建 = 幂等 |
| 查 | `search.py` | 锁菜 + 类型过滤 + 向量检索 + trace |
| 拼 | `ask.py` | 受控生成：只准依据证据回答 |

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 本地 Ollama 备好两个模型（服务需在 localhost:11434 运行）
ollama pull bge-m3
ollama pull qwen3:4b

# 3. 备菜（入库，跑一次；数据更新时重跑，幂等不翻倍）
python prep/build_menu_chunks.py        # 期望：总块数: 61
python prep/convert_restaurant.py       # 期望：质检全绿 ✅
python prep/build_restaurant_chunks.py  # 期望：总块数: 6
python prep/build_index.py              # 期望：双仓 61 + 6

# 4. 点菜（日常入口）
python search.py "锅包肉怎么做"            # 锁菜版
python search.py "有什么清淡的汤"          # 裸搜版
python search.py "有什么下饭的" --type 素菜 # 类型过滤版
python ask.py "锅包肉怎么做"               # 受控生成答案
python ask.py "佛跳墙怎么做"               # 防幻觉验收：应答"菜单里没有"
```

## 设计要点

### 两仓通用

- **一块自包含，永不跨界**：一道菜一块、一分店一块（= drug_rag"chunk 永不跨章节"）；
  连锁分店行写全公共信息，单独被检索到也能看全
- **字段分家（下游倒推）**：可枚举短字段（菜名/类型/口味/品类/商圈/人均…）→ metadata（过滤+锁定用）；
  描述性长文本（做法/招牌菜/避雷点…）→ 正文（向量搜索用）
- **暗桩前缀**：正文首行缝标签关键词（"锅包肉 荤菜 酸甜"、"大富烧鸡（坂田店） 烧鸡 龙华 …"）——
  标签词若只躺 metadata，向量搜不到（学 drug_rag 摘要卡药名前缀）
- **整仓删除重建 = 幂等**：曾用"先删后存"按 id 删，id 规则一改旧块成僵尸、仓库翻倍成 122，
  验收块数当场逮住后改为整仓重建（学 drug_rag 的先删后存，但 mini 规模用更稳的变体）

### 菜品仓

- **锁菜**：`菜名 in 问题` 唯一命中才锁，filter={菜名:…}，防"锅包豆腐顶替锅包肉"（= drug_rag 锁药）

### 饭店仓

- **md → json + 质检台**：转换即对账（必填/人均纯数字/商圈单值/占位符检测），
  垃圾挡在门外——真逮住过占位符没换、商圈多值两案
- **一分店一行**：= drug_rag"聚合键是 doc_id 不是药名"——连锁同名分店是不同实体，
  id 靠分店区分（restaurant:大富烧鸡:锦绣店）
- **人均存 int**：Chroma 支持数值比较，将来可 `filter 人均 ≤ 50`
- **商圈单值**：Chroma metadata 不支持列表，多值 = 过滤废掉

### 生成

- **受控生成**：系统提示词规定只准依据证据回答，证据不足必须明说——问菜单上没有的菜不编造
- **LLM 一行切换**：`ask.py` 的 `PROVIDER = "ollama"` 改成 `"dashscope"` 即换供应商，
  需装 `langchain-openai` + 环境变量 `DASHSCOPE_API_KEY`

## 已知局限与升级路线

- 锁菜不认别名：问"孜然牛肉怎么做"锁不上"孜盐牛肉"（备注里写着是同义词）
  → 升级方向：马甲表扩到备注别名（= drug_rag 的3马甲）
- search/ask 还不认识饭店仓 → 下一步：选仓能力（问做法查菜品仓、找店查饭店仓）+
  人均数值过滤 + 连锁多命中处理（= drug_rag ambiguous 场景）
- 拒答时"来源"栏会把问题里的菜名当来源写 → 来源强制从证据 metadata 生成
- 无意图路由/多路召回/重排/安全审查 → 都是 drug_rag 已拆过的豪华配置，按需搬

## 向 drug_rag2.0 致敬（同构对照）

| drug_rag2.0 | FoodRAG |
|---|---|
| 说明书 PDF 138 块 | 菜单 json 61 块 + 饭店 md 6 块 |
| 章节标签 section | 类型/品类标签 |
| catalog 锁药（3马甲） | 锁菜（菜名马甲） |
| 药名 in 问题 锁药 | 菜名/店名 in 问题 锁定 |
| PDF→JSONL 解析入库 | md→json 质检台→JSONL |
| 摘要两跳 / 多路召回 / 重排 | 纯向量一路 |
| qa_chain 五段式 + 安全链 | 受控生成 + 来源标注 |
| ~6200 行 | ~500 行 |

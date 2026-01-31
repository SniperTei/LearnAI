# Text2SQL 智能查询助手

一个基于大语言模型的 Text2SQL 演示项目，让用户用自然语言查询数据库，无需掌握 SQL 语法。

## 项目特性

- 🤖 **智能 SQL 生成**: 使用 GPT-4 将自然语言转换为 SQL
- 📊 **保险场景示例**: 包含完整的保险业务数据库（客户、保单、理赔、产品）
- 🎨 **友好的 Web 界面**: 基于 Streamlit 的交互式查询界面
- 🔧 **两种实现方式**:
  - LangChain SQL Agent（快速上手）
  - 自定义 Prompt（完全控制）
- 📝 **结果解释**: AI 自动解释查询结果

## 项目结构

```
text2sql/
├── README.md              # 项目说明文档
├── requirements.txt       # Python 依赖
├── .env                  # 环境变量配置（需自行创建）
├── init_database.py      # 数据库初始化脚本
├── text2sql_agent.py     # LangChain SQL Agent 实现
├── text2sql_custom.py    # 自定义 Prompt 实现
└── app.py               # Streamlit Web 应用
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 OpenAI API Key

在项目目录创建 `.env` 文件：

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 初始化数据库

```bash
python init_database.py
```

这将创建 `insurance.db` 数据库文件，包含以下表：
- `customers`: 客户信息（20条示例数据）
- `products`: 保险产品（5个产品）
- `policies`: 保单记录（约40条）
- `claims`: 理赔记录（30条）

### 4. 运行演示

#### 方式一：使用 LangChain Agent（推荐新手）

```bash
python text2sql_agent.py
```

#### 方式二：使用自定义 Prompt（推荐进阶）

```bash
python text2sql_custom.py
```

#### 方式三：启动 Web 界面（推荐）

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`

## 使用示例

### 通过代码

```python
from text2sql_agent import Text2SQLAgent

# 创建 Agent
agent = Text2SQLAgent()

# 自然语言查询
answer = agent.query("有多少个客户？")
print(answer)
```

### 通过 Web 界面

1. 在输入框中输入问题，如："查询年龄在30到40岁之间的女性客户"
2. 点击"查询"按钮
3. 查看 AI 生成的 SQL 和查询结果
4. 查看 AI 的结果解释

## 支持的查询类型

- ✅ **简单查询**: "有多少个客户？"
- ✅ **条件查询**: "查询年龄大于30的客户"
- ✅ **聚合统计**: "统计每个城市的客户数量"
- ✅ **排序查询**: "查询保费最高的5个保单"
- ✅ **多表关联**: "查询保单对应的客户信息"
- ✅ **复杂计算**: "北京地区的平均保费是多少？"

## 示例问题

- 有多少个客户？
- 查询年龄在30到40岁之间的女性客户
- 统计每个产品的保单数量
- 查询保费最高的5个保单
- 北京地区的客户平均年龄是多少？
- 查询状态为'已批准'的理赔记录
- 赔付率最高的3个产品是什么？

## 技术栈

- **LLM**: GPT-4o (可替换为其他支持 OpenAI API 的模型)
- **框架**: LangChain 0.1.0
- **数据库**: SQLite
- **Web**: Streamlit 1.31.0

## 代码说明

### 1. init_database.py

创建保险场景的示例数据库，包含：
- 数据库表结构定义
- 示例数据生成（客户、产品、保单、理赔）

### 2. text2sql_agent.py

使用 LangChain 的 `create_sql_agent` 实现：
- 自动处理数据库连接
- 内置 SQL 执行工具
- 零样本推理能力
- 适合快速原型开发

### 3. text2sql_custom.py

完全自定义的 Prompt 工程实现：
- Few-shot Learning（提供示例）
- 自定义 System Prompt
- SQL 清洗和验证
- 带解释的查询功能
- 适合生产环境优化

### 4. app.py

Streamlit Web 应用：
- 友好的用户界面
- 实时查询历史
- 数据表展示
- CSV 结果下载
- AI 结果解释

## 优化方向

### 提高准确率

1. **Few-shot Learning**: 在 Prompt 中提供更多示例
2. **Schema 过滤**: 只包含相关的表结构
3. **思维链**: 让模型先思考再生成
4. **自愈机制**: 捕获错误并让模型修正

### 性能优化

1. **连接池**: 复用数据库连接
2. **查询缓存**: 缓存常见查询
3. **异步处理**: 使用 async 提升响应速度
4. **批量查询**: 支持一次查询多个问题

### 安全性

1. **SQL 注入防护**: 验证生成的 SQL
2. **权限控制**: 限制可查询的表和字段
3. **查询超时**: 防止长时间运行
4. **结果限制**: 默认限制返回数量

### 生产部署

1. **切换到生产数据库**: PostgreSQL / MySQL
2. **用户认证**: 添加登录系统
3. **审计日志**: 记录所有查询
4. **监控告警**: 监控 API 调用和错误率

## 常见问题

### Q: 如何更换数据库？

A: 修改连接字符串：
```python
# SQLite
db = SQLDatabase.from_uri("sqlite:///insurance.db")

# MySQL
db = SQLDatabase.from_uri("mysql+pymysql://user:pass@localhost/dbname")

# PostgreSQL
db = SQLDatabase.from_uri("postgresql://user:pass@localhost/dbname")
```

### Q: 如何降低 API 成本？

A:
1. 使用 `gpt-4o-mini` 替代 `gpt-4o`
2. 添加查询缓存
3. 减少返回的 schema 信息

### Q: 查询结果不准确怎么办？

A:
1. 在 Prompt 中添加更多 Few-shot 示例
2. 优化表结构描述（添加注释）
3. 添加业务术语映射
4. 实现自愈机制（捕获错误并重试）

### Q: 支持中文查询吗？

A: 完全支持！项目默认配置就是中文，数据库表名和字段名都可以用中文提问。

## 学习资源

- [LangChain SQL Documentation](https://python.langchain.com/docs/use_cases/sql)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Spider Dataset (Text2SQL 基准测试)](https://yale-lily.github.io/spider)

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。

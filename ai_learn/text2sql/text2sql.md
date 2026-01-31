# Lesson 08: Text2SQL - 自然语言到SQL的智能转换

## 课程概述

本课程将教你如何使用大语言模型(LLM)构建Text2SQL应用，让用户能够用自然语言查询数据库，无需掌握SQL语法。这是企业数据分析和自助式报表开发的核心技术。

---

## 学习内容

### 1. 自助式数据报表开发
- **痛点**: 业务人员需要数据，但不懂SQL，依赖IT部门开发报表
- **解决方案**: 通过自然语言直接查询数据库，即时获取数据
- **价值**:
  - 降低数据分析门槛
  - 提高业务效率
  - 减少技术人员工作量

### 2. Text to SQL 核心技术
- **定义**: 将自然语言问题自动转换为SQL查询语句
- **示例**:
  ```
  输入: "查询上个月销售额最高的前10个产品"
  输出: SELECT product_name, SUM(amount) as total
        FROM sales
        WHERE date >= '2024-12-01'
        GROUP BY product_name
        ORDER BY total DESC
        LIMIT 10
  ```
- **技术挑战**:
  - 理解用户意图
  - 识别数据库schema
  - 处理复杂的业务逻辑
  - SQL语法准确性

### 3. LLM模型选择
- **商业模型**:
  - GPT-4 / GPT-4o: SQL生成能力强，适合生产环境
  - Claude 3.5 Sonnet: 在代码和结构化查询上表现出色
  - Gemini Pro: Google生态集成良好

- **开源模型**:
  - CodeLlama: 专门针对代码生成优化
  - SQLCoder: 专注于SQL任务的专用模型
  - DeepSeek-Coder: 性价比高的中文友好的代码模型

- **选择标准**:
  - SQL生成准确率
  - 上下文理解能力
  - 成本考量
  - 数据隐私要求

### 4. Function Call (函数调用)
- **作用**: 让LLM能够调用外部工具执行SQL
- **流程**:
  1. 用户输入自然语言问题
  2. LLM理解意图并决定调用SQL执行函数
  3. 执行SQL查询数据库
  4. 将结果返回给LLM
  5. LLM用自然语言总结结果

- **优势**:
  - 安全可控
  - 可以添加权限验证
  - 支持多步骤复杂查询

### 5. 搭建SQL Copilot
- **定位**: SQL编程助手，辅助开发者
- **功能**:
  - SQL自动补全
  - 查询优化建议
  - 错误诊断和修复
  - 自然语言解释SQL
  - 复杂查询生成

- **开发要点**:
  - 连接数据库元信息
  - 上下文管理(表结构、历史查询)
  - Prompt工程优化

### 6. LangChain中的SQL Agent
- **SQLDatabaseToolkit**: LangChain提供的SQL工具集
  - 数据库连接
  - Schema查询
  - SQL执行
  - 结果验证

- **Agent类型**:
  - **ZERO_SHOT_REACT_DESCRIPTION**: 零样本推理
  - **SQL_AGENT**: 专门针对SQL优化

- **实现流程**:
  ```python
  from langchain.agents import create_sql_agent
  from langchain_community.utilities import SQLDatabase

  # 连接数据库
  db = SQLDatabase.from_uri("sqlite:///chinook.db")

  # 创建Agent
  agent = create_sql_agent(
      llm=llm,
      toolkit=db.get_sql_toolkit(),
      verbose=True
  )
  ```

### 7. 自定义LLM + Prompt工程
- **核心Prompt结构**:
  ```
  你是一个SQL专家。根据以下信息生成SQL查询:

  数据库Schema:
  {schema_info}

  用户问题:
  {user_question}

  注意事项:
  - 只返回SQL语句，不要解释
  - 使用合适的表连接
  - 添加必要的条件过滤
  - 限制结果数量避免返回过多数据

  SQL:
  ```

- **优化技巧**:
  - Few-shot learning: 提供示例
  - Schema检索: 只包含相关表结构
  - 思维链: 让模型先思考再生成
  - 迭代优化: 根据错误反馈修正

### 8. 实战案例: 保险场景SQL Copilot

- **业务场景**:
  - 保险公司的数据分析需求
  - 保单查询、理赔统计、客户分析等

- **核心表结构**:
  - `policies`: 保单信息
  - `customers`: 客户信息
  - `claims`: 理赔记录
  - `products`: 保险产品

- **典型查询**:
  - "本月新增保单数量"
  - "赔付率最高的险种"
  - "30-40岁女性客户的平均保费"
  - "逾期未缴费保单统计"

- **实现要点**:
  - 业务术语映射
  - 复杂聚合查询
  - 多表关联优化
  - 结果可视化

---

## 技术栈建议

- **LLM**: OpenAI GPT-4o / Anthropic Claude 3.5 Sonnet
- **框架**: LangChain / LlamaIndex
- **数据库**: PostgreSQL / MySQL / SQLite
- **前端**: Streamlit / Gradio (快速原型)

---

## 学习路径

1. **基础**: 理解Text2SQL原理和挑战
2. **实践**: 使用LangChain SQL Toolkit快速搭建原型
3. **深入**: 自定义Prompt优化准确率
4. **实战**: 完成保险场景Copilot项目
5. **优化**: 处理复杂查询和边缘case

---

## 参考资料

- LangChain SQL Documentation
- Spider Dataset (Text2SQL基准数据集)
- "How to Build a SQL Agent with LangChain"

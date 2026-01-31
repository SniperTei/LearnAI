"""
Text2SQL Streamlit 演示应用
一个友好的 Web 界面，让用户用自然语言查询数据库
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="Text2SQL 智能查询助手",
    page_icon="🤖",
    layout="wide"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sql-box {
        background-color: #2E3440;
        color: #88C0D0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化 session state"""
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


@st.cache_resource
def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect('insurance.db')
    conn.row_factory = sqlite3.Row
    return conn


def get_schema():
    """获取数据库 schema"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    schema_info = "数据库包含以下表:\n\n"

    for table in tables:
        schema_info += f"### 表: {table}\n"

        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()

        schema_info += "| 列名 | 类型 |\n|------|------|\n"
        for col in columns:
            schema_info += f"| {col[1]} | {col[2]} |\n"
        schema_info += "\n"

    conn.close()
    return schema_info


def init_llm():
    """初始化 LLM"""
    if not os.getenv('OPENAI_API_KEY'):
        st.error("❌ 请先设置 OPENAI_API_KEY 环境变量")
        st.info("💡 在项目目录创建 .env 文件，添加: OPENAI_API_KEY=your_key_here")
        return None

    return ChatOpenAI(
        model='gpt-4o',
        temperature=0
    )


def create_prompt_template():
    """创建 Prompt 模板"""
    examples = [
        {
            "question": "有多少个客户？",
            "sql": "SELECT COUNT(*) as total FROM customers;"
        },
        {
            "question": "查询年龄在30到40岁之间的女性客户",
            "sql": "SELECT * FROM customers WHERE age BETWEEN 30 AND 40 AND gender = '女';"
        },
        {
            "question": "统计每个产品的保单数量",
            "sql": "SELECT p.product_name, COUNT(po.policy_id) as policy_count FROM products p LEFT JOIN policies po ON p.product_id = po.product_id GROUP BY p.product_id, p.product_name;"
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}\nSQL: {sql}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个 SQL 专家。根据用户的问题和数据库 schema，生成准确的 SQLite 查询语句。

要求：
1. 只返回 SQL 语句，不要解释
2. 使用合适的表连接
3. 添加必要的条件过滤
4. 限制结果数量避免返回过多数据（使用 LIMIT）
5. 确保语法正确

数据库 Schema:
{schema}"""),
        few_shot_prompt,
        ("human", "{question}")
    ])

    return prompt


def generate_sql(llm, prompt, schema, question):
    """生成 SQL"""
    messages = prompt.format_messages(
        schema=schema,
        question=question
    )

    response = llm.invoke(messages)
    sql = response.content.strip()

    # 清理 markdown 标记
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


def execute_query(sql):
    """执行 SQL 查询"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]
    except Exception as e:
        return {"error": str(e)}


def explain_result(llm, question, sql, results):
    """让 LLM 解释查询结果"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个数据分析助手。请用简洁友好的中文解释查询结果，不超过2句话。"),
        ("human", "问题: {question}\nSQL: {sql}\n结果: {results}\n请解释这个结果。")
    ])

    messages = prompt.format_messages(
        question=question,
        sql=sql,
        results=str(results)[:1000]  # 限制长度
    )

    return llm.invoke(messages).content


def main():
    """主应用"""
    init_session_state()

    # 标题
    st.markdown('<h1 class="main-header">🤖 Text2SQL 智能查询助手</h1>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")

        # API Key 检查
        if not os.getenv('OPENAI_API_KEY'):
            st.error("❌ 未检测到 OPENAI_API_KEY")
            st.info("""
            请在项目目录创建 .env 文件，添加:
            ```
            OPENAI_API_KEY=your_key_here
            ```
            """)
            st.stop()

        st.success("✅ API Key 已配置")

        st.divider()

        # 数据库信息
        st.subheader("📊 数据库信息")
        if st.button("刷新数据库信息"):
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            st.metric("客户数", cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM policies")
            st.metric("保单数", cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM claims")
            st.metric("理赔数", cursor.fetchone()[0])
            conn.close()

        st.divider()

        # Schema 查看
        with st.expander("📋 查看数据库表结构"):
            st.markdown(get_schema())

        st.divider()

        # 查询历史
        st.subheader("📜 查询历史")
        if len(st.session_state.query_history) > 0:
            for i, (q, s) in enumerate(st.session_state.query_history[-5:], 1):
                st.text(f"{i}. {q}")
        else:
            st.info("暂无查询历史")

    # 主界面
    st.header("💬 自然语言查询")

    # 示例问题
    example_questions = [
        "有多少个客户？",
        "查询年龄在30到40岁之间的女性客户",
        "统计每个产品的保单数量",
        "查询保费最高的5个保单",
        "北京地区的客户平均年龄是多少？",
        "查询状态为'已批准'的理赔记录",
    ]

    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_input(
            "输入你的问题:",
            placeholder="例如：查询所有年龄大于30岁的客户",
            label_visibility="collapsed"
        )

    with col2:
        st.write("")  # 对齐
        st.write("")
        random_example = st.selectbox("或选择示例:", [""] + example_questions, label_visibility="collapsed")

    # 如果选择了示例，填充到输入框
    if random_example and random_example != question:
        question = random_example

    # 查询按钮
    submit_button = st.button("🔍 查询", type="primary", use_container_width=True)

    if submit_button and question:
        # 初始化 LLM
        with st.spinner("正在初始化..."):
            llm = init_llm()
            if not llm:
                st.stop()

        # 获取 schema
        schema = get_schema()

        # 创建 prompt
        prompt = create_prompt_template()

        # 生成 SQL
        with st.spinner("正在生成 SQL..."):
            sql = generate_sql(llm, prompt, schema, question)

        # 显示生成的 SQL
        st.subheader("📝 生成的 SQL")
        st.code(sql, language="sql", line_numbers=True)

        # 执行查询
        with st.spinner("正在执行查询..."):
            results = execute_query(sql)

        # 显示结果
        st.subheader("📊 查询结果")

        if isinstance(results, dict) and "error" in results:
            st.error(f"❌ 查询出错: {results['error']}")
        else:
            if len(results) == 0:
                st.info("📭 查询结果为空")
            else:
                # 显示数据表格
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # 显示统计信息
                st.info(f"✅ 查询成功！共 {len(results)} 条记录")

                # 让 LLM 解释结果
                with st.spinner("正在生成结果解释..."):
                    explanation = explain_result(llm, question, sql, results)

                st.subheader("💡 结果解释")
                st.success(explanation)

                # 添加到历史
                st.session_state.query_history.append((question, sql))

                # 下载按钮
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 下载结果 (CSV)",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )

    # 使用说明
    st.divider()
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 如何使用本应用

        1. **输入问题**: 在上方输入框中输入你的问题，或从下拉菜单选择示例问题
        2. **生成 SQL**: 点击"查询"按钮，AI 会自动将你的问题转换为 SQL 语句
        3. **查看结果**: 查看 AI 生成的 SQL 和查询结果
        4. **结果解释**: AI 会用自然语言解释查询结果

        ### 支持的查询类型

        - 简单查询: "有多少个客户？"
        - 条件查询: "查询年龄大于30的客户"
        - 聚合统计: "统计每个城市的客户数量"
        - 排序查询: "查询保费最高的5个保单"
        - 多表关联: "查询保单对应的客户信息"

        ### 数据库表说明

        - **customers**: 客户信息表（姓名、年龄、性别、城市等）
        - **products**: 保险产品表（产品名称、类型、保费范围等）
        - **policies**: 保单表（客户、产品、日期、保费、状态等）
        - **claims**: 理赔记录表（保单、日期、金额、状态等）
        """)


if __name__ == "__main__":
    main()

"""
Text2SQL Streamlit 演示应用 - 使用本地 Ollama
一个友好的 Web 界面，让用户用自然语言查询数据库
"""

import streamlit as st
import pandas as pd
import sqlite3
import requests
import json

# Ollama 配置
OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "deepseek-r1:7b"  # 或使用其他模型如 "qwen2.5:7b"

# 页面配置
st.set_page_config(
    page_title="Text2SQL 智能查询助手 (Ollama)",
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
    conn = sqlite3.connect('insurance.db', check_same_thread=False)
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

    # Don't close the connection - it's cached by @st.cache_resource
    return schema_info


def check_ollama():
    """检查 Ollama 是否运行"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def call_ollama(prompt: str, system_prompt: str = "") -> str:
    """
    调用 Ollama API

    Args:
        prompt: 用户提示
        system_prompt: 系统提示

    Returns:
        模型响应
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # 降低温度以提高 SQL 准确性
                    "num_predict": 500
                }
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "")
    except Exception as e:
        return f"Error: {str(e)}"


def generate_sql(schema: str, question: str) -> str:
    """使用 Ollama 生成 SQL"""

    # 示例
    examples = """
示例1:
问题: 有多少个客户？
SQL: SELECT COUNT(*) as total FROM customers;

示例2:
问题: 查询年龄在30到40岁之间的女性客户
SQL: SELECT * FROM customers WHERE age BETWEEN 30 AND 40 AND gender = '女';

示例3:
问题: 统计每个产品的保单数量
SQL: SELECT p.product_name, COUNT(po.policy_id) as policy_count FROM products p LEFT JOIN policies po ON p.product_id = po.product_id GROUP BY p.product_id, p.product_name;
"""

    system_prompt = """你是一个 SQL 专家。根据用户的问题和数据库 schema，生成准确的 SQLite 查询语句。

要求：
1. 只返回 SQL 语句，不要解释
2. 使用合适的表连接（JOIN）
3. 添加必要的条件过滤（WHERE）
4. 限制结果数量避免返回过多数据（使用 LIMIT，建议限制100条）
5. 确保语法正确
6. 不要使用 markdown 格式（不要用 ```sql 包裹）
7. 直接输出 SQL 语句"""

    prompt = f"""数据库 Schema:
{schema}

{examples}

请为以下问题生成 SQL 查询：

问题: {question}

SQL:"""

    response = call_ollama(prompt, system_prompt)

    # 清理响应
    sql = response.strip()

    # 移除可能的 markdown 标记
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    # 移除可能的 "SQL:" 前缀
    if sql.startswith("SQL:"):
        sql = sql[4:]

    return sql.strip()


def explain_result(question: str, sql: str, results: str) -> str:
    """让 Ollama 解释查询结果"""
    system_prompt = "你是一个数据分析助手。请用简洁友好的中文解释查询结果，不超过2句话。"

    prompt = f"""问题: {question}
SQL: {sql}
结果: {results[:1000]}

请解释这个结果。"""

    return call_ollama(prompt, system_prompt)


def execute_query(sql):
    """执行 SQL 查询"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        # Don't close the connection - it's cached by @st.cache_resource
        return [dict(row) for row in rows]
    except Exception as e:
        return {"error": str(e)}


def main():
    """主应用"""
    init_session_state()

    # 标题
    st.markdown('<h1 class="main-header">🤖 Text2SQL 智能查询助手 (Ollama)</h1>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")

        # Ollama 状态检查
        if check_ollama():
            st.success(f"✅ Ollama 运行中")
            st.info(f"📝 模型: {CHAT_MODEL}")
        else:
            st.error("❌ 无法连接到 Ollama")
            st.info("""
            请确保 Ollama 正在运行:
            1. 安装 Ollama: https://ollama.com
            2. 启动服务: ollama serve
            3. 拉取模型: ollama pull deepseek-r1:7b
            """)
            st.stop()

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
            # Don't close the connection - it's cached by @st.cache_resource

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
        # 获取 schema
        schema = get_schema()

        # 生成 SQL
        with st.spinner("正在生成 SQL..."):
            sql = generate_sql(schema, question)

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
                    explanation = explain_result(question, sql, str(results))

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

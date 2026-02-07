"""
Ollama Text2SQL Streamlit Demo App
A friendly web interface to query SQLite database with natural language using Ollama
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Ollama Text2SQL Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
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
    """Initialize session state"""
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


@st.cache_resource
def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect('../text2sql_sqlite_refined/insurance.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn


def get_schema():
    """Get database schema"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    schema_info = "数据库包含以下表:\n\n"

    for table in tables:
        schema_info += f"### 表: {table}\n"

        # Get column information for each table
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()

        schema_info += "| 列名 | 类型 |\n|------|------|\n"
        for col in columns:
            schema_info += f"| {col[1]} | {col[2]} |\n"
        schema_info += "\n"

    # Don't close the connection - it's cached by @st.cache_resource
    return schema_info


def init_llm():
    """Initialize Ollama LLM"""
    # Allow user to select model
    available_models = ["llama3", "mistral", "codellama", "phi"]
    selected_model = st.sidebar.selectbox("Select Ollama Model", available_models, index=0)
    
    try:
        llm = Ollama(model=selected_model, base_url="http://localhost:11434")
        return llm
    except Exception as e:
        st.error(f"❌ Error connecting to Ollama: {e}")
        st.info("💡 Make sure Ollama is running on your system")
        return None


def create_prompt_template():
    """Create Prompt template"""
    examples = [
        {
            "question": "How many customers are there?",
            "sql": "SELECT COUNT(*) as total FROM customers;"
        },
        {
            "question": "Find female customers between 30 and 40 years old",
            "sql": "SELECT * FROM customers WHERE age BETWEEN 30 AND 40 AND gender = 'Female';"
        },
        {
            "question": "Count policies for each product",
            "sql": "SELECT p.product_name, COUNT(po.policy_id) as policy_count FROM products p LEFT JOIN policies po ON p.product_id = po.product_id GROUP BY p.product_id, p.product_name;"
        },
        {
            "question": "Show customers in New York",
            "sql": "SELECT * FROM customers WHERE city = 'New York';"
        },
        {
            "question": "Get the highest premium amount",
            "sql": "SELECT MAX(premium_amount) as highest_premium FROM policies;"
        }
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}\nSQL: {sql}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Based on the user's question and database schema, generate an accurate SQLite query.

Requirements:
1. Return only the SQL statement, no explanation
2. Use proper table joins when needed
3. Add necessary filtering conditions
4. Limit results to avoid returning too much data (use LIMIT)
5. Ensure correct syntax for SQLite

Database Schema:
{schema}"""),
        few_shot_prompt,
        ("human", "{question}")
    ])

    return prompt


def generate_sql(llm, prompt, schema, question):
    """Generate SQL"""
    messages = prompt.format_messages(
        schema=schema,
        question=question
    )

    # Convert to string format for Ollama
    formatted_prompt = f"System: {messages[0].content}\nHuman: {messages[1].content}"
    
    response = llm.invoke(formatted_prompt)
    sql = response.strip()

    # Clean up markdown formatting
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


def execute_query(sql):
    """Execute SQL query"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Execute the query
        cursor.execute(sql)
        
        # Fetch all results
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        results = [dict(row) for row in rows]
            
        return results
    except Exception as e:
        return {"error": str(e)}


def explain_result(llm, question, sql, results):
    """Have LLM explain query results"""
    prompt_text = f"""You are a data analysis assistant. Explain the query results in simple, friendly Chinese in no more than 2 sentences.
    
Question: {question}
SQL: {sql}
Results: {str(results)[:1000]}  # Limit length

Please explain these results."""
    
    return llm.invoke(prompt_text)


def main():
    """Main application"""
    init_session_state()

    # Title
    st.markdown('<h1 class="main-header">🤖 Ollama Text2SQL Assistant</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Ollama connection check
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                st.success("✅ Ollama is running")
            else:
                st.error("❌ Ollama is not responding")
        except:
            st.error("❌ Cannot connect to Ollama")
            st.info("💡 Make sure Ollama is running on your system")
            st.stop()

        st.divider()

        # Database info
        st.subheader("📊 Database Info")
        if st.button("Refresh database info"):
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM customers")
            st.metric("Customers", cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM policies")
            st.metric("Policies", cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM claims")
            st.metric("Claims", cursor.fetchone()[0])
            # Don't close the connection - it's cached by @st.cache_resource

        st.divider()

        # Schema view
        with st.expander("📋 View Database Schema"):
            st.markdown(get_schema())

        st.divider()

        # Query history
        st.subheader("📜 Query History")
        if len(st.session_state.query_history) > 0:
            for i, (q, s) in enumerate(st.session_state.query_history[-5:], 1):
                st.text(f"{i}. {q}")
        else:
            st.info("No query history")

    # Main interface
    st.header("💬 Natural Language Query")

    # Example questions
    example_questions = [
        "How many customers are there?",
        "Find female customers between 30 and 40 years old",
        "Count policies for each product",
        "Show customers in New York",
        "Get the highest premium amount",
        "Show approved claims over $10,000",
        "List all active policies"
    ]

    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="Example: Find all customers older than 30",
            label_visibility="collapsed"
        )

    with col2:
        st.write("")  # Alignment
        st.write("")
        random_example = st.selectbox("Or select an example:", [""] + example_questions, label_visibility="collapsed")

    # If example selected, fill the input box
    if random_example and random_example != question:
        question = random_example

    # Query button
    submit_button = st.button("🔍 Query", type="primary", use_container_width=True)

    if submit_button and question:
        # Initialize Ollama LLM
        with st.spinner("Initializing Ollama..."):
            llm = init_llm()
            if not llm:
                st.stop()

        # Get schema
        schema = get_schema()

        # Create prompt
        prompt = create_prompt_template()

        # Generate SQL
        with st.spinner("Generating SQL with Ollama..."):
            sql = generate_sql(llm, prompt, schema, question)

        # Display generated SQL
        st.subheader("📝 Generated SQL")
        st.code(sql, language="sql", line_numbers=True)

        # Execute query
        with st.spinner("Executing query..."):
            results = execute_query(sql)

        # Display results
        st.subheader("📊 Query Results")

        if isinstance(results, dict) and "error" in results:
            st.error(f"❌ Query error: {results['error']}")
        else:
            if len(results) == 0:
                st.info("NullOrEmpty Query returned no results")
            else:
                # Show data table
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Show stats
                st.info(f"✅ Query successful! {len(results)} records returned")

                # Have LLM explain results
                with st.spinner("Generating result explanation..."):
                    explanation = explain_result(llm, question, sql, results)

                st.subheader("💡 Result Explanation")
                st.success(explanation)

                # Add to history
                st.session_state.query_history.append((question, sql))

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )

    # Usage instructions
    st.divider()
    with st.expander("📖 Usage Instructions"):
        st.markdown("""
        ### How to Use This App

        1. **Enter Question**: Type your question in the input box above, or select from the dropdown examples
        2. **Generate SQL**: Click the "Query" button, and Ollama will convert your question to SQL
        3. **View Results**: See the SQL generated by Ollama and the query results
        4. **Result Explanation**: Ollama will explain the query results in natural language

        ### Supported Query Types

        - Simple queries: "How many customers?"
        - Conditional queries: "Find customers older than 30"
        - Aggregation: "Count customers by city"
        - Sorting: "Find top 5 highest premium policies"
        - Joins: "Find policy details for customers"

        ### Database Tables

        - **customers**: Customer information (name, age, gender, city, etc.)
        - **products**: Insurance products (name, type, premium range, etc.)
        - **policies**: Policy information (customer, product, date, premium, etc.)
        - **claims**: Claims records (policy, date, amount, status, etc.)
        """)


if __name__ == "__main__":
    main()
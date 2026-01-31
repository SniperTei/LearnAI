"""
Text2SQL 自定义版本 - 使用自定义 Prompt
这个版本展示如何完全控制 Prompt 工程来优化 Text2SQL 效果
"""

import os
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()


class CustomText2SQL:
    """自定义 Prompt 的 Text2SQL 实现"""

    def __init__(self, db_path='insurance.db'):
        """
        初始化自定义 Text2SQL

        Args:
            db_path: 数据库文件路径
        """
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # 返回字典格式

        # 获取数据库 schema
        self.schema = self._get_schema()

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model='gpt-4o',
            temperature=0
        )

        # 创建 Prompt 模板
        self.prompt = self._create_prompt()

        print("✓ 自定义 Text2SQL 初始化完成")

    def _get_schema(self):
        """获取数据库 schema 信息"""
        cursor = self.conn.cursor()

        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_info = "数据库表结构:\n\n"

        for table in tables:
            table_name = table[0]
            schema_info += f"## 表: {table_name}\n"

            # 获取表的列信息
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            schema_info += "列:\n"
            for col in columns:
                schema_info += f"  - {col[1]} ({col[2]})\n"

            schema_info += "\n"

        return schema_info

    def _create_prompt(self):
        """创建自定义 Prompt 模板"""

        # Few-shot 示例
        examples = [
            {
                "question": "有多少个客户？",
                "sql": "SELECT COUNT(*) FROM customers;"
            },
            {
                "question": "查询年龄在30到40岁之间的女性客户",
                "sql": "SELECT * FROM customers WHERE age BETWEEN 30 AND 40 AND gender = '女';"
            },
            {
                "question": "统计每个产品的保单数量",
                "sql": "SELECT p.product_name, COUNT(po.policy_id) as policy_count FROM products p LEFT JOIN policies po ON p.product_id = po.product_id GROUP BY p.product_id, p.product_name;"
            },
            {
                "question": "查询北京地区的平均保费",
                "sql": "SELECT AVG(premium) as avg_premium FROM policies po JOIN customers c ON po.customer_id = c.customer_id WHERE c.city = '北京';"
            },
        ]

        # 创建 Few-shot 提示模板
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}\nSQL: {sql}")
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        # 完整的 Prompt 模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个 SQL 专家。根据用户的问题和数据库 schema，生成准确的 SQL 查询语句。

要求：
1. 只返回 SQL 语句，不要解释
2. 使用合适的表连接
3. 添加必要的条件过滤
4. 限制结果数量避免返回过多数据（如果适用）
5. 确保语法正确
6. 使用 SQLite 语法

数据库 Schema:
{schema}"""),
            few_shot_prompt,
            ("human", "{question}")
        ])

        return prompt

    def generate_sql(self, question):
        """
        根据自然语言问题生成 SQL

        Args:
            question: 自然语言问题

        Returns:
            生成的 SQL 语句
        """
        # 格式化 prompt
        messages = self.prompt.format_messages(
            schema=self.schema,
            question=question
        )

        # 调用 LLM
        response = self.llm.invoke(messages)

        # 提取 SQL（去除可能的 markdown 标记）
        sql = response.content.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()

    def execute_sql(self, sql):
        """
        执行 SQL 查询

        Args:
            sql: SQL 语句

        Returns:
            查询结果（字典列表格式）
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

            # 转换为字典列表
            results = [dict(row) for row in rows]
            return results
        except Exception as e:
            return {"error": str(e)}

    def query(self, question):
        """
        完整的查询流程：生成 SQL -> 执行 -> 返回结果

        Args:
            question: 自然语言问题

        Returns:
            查询结果
        """
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")

        # 生成 SQL
        print("\n正在生成 SQL...")
        sql = self.generate_sql(question)
        print(f"生成的 SQL:\n{sql}\n")

        # 执行 SQL
        print("正在执行 SQL...")
        results = self.execute_sql(sql)

        # 显示结果
        if isinstance(results, list):
            if len(results) == 0:
                print("查询结果: 无数据")
            else:
                print(f"查询结果 ({len(results)} 条):")
                for i, row in enumerate(results, 1):
                    print(f"  {i}. {row}")
        else:
            print(f"查询结果: {results}")

        return results

    def query_with_explanation(self, question):
        """
        带解释的查询 - 不仅返回结果，还让 LLM 解释结果

        Args:
            question: 自然语言问题

        Returns:
            SQL + 结果 + 自然语言解释
        """
        # 生成并执行 SQL
        sql = self.generate_sql(question)
        results = self.execute_sql(sql)

        # 让 LLM 解释结果
        explanation_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个数据分析助手。请用简洁的中文解释查询结果。"),
            ("human", "问题: {question}\nSQL: {sql}\n结果: {results}\n请解释这个结果。")
        ])

        messages = explanation_prompt.format_messages(
            question=question,
            sql=sql,
            results=str(results)
        )

        explanation = self.llm.invoke(messages).content

        return {
            "sql": sql,
            "results": results,
            "explanation": explanation
        }

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


def main():
    """演示自定义 Text2SQL 的使用"""

    print("="*60)
    print("自定义 Text2SQL 演示")
    print("="*60)

    text2sql = CustomText2SQL()

    # 示例查询
    questions = [
        "有多少个客户？",
        "查询年龄大于50岁的客户",
        "统计每个城市的客户数量",
        "查询保费最高的5个保单",
    ]

    for question in questions:
        text2sql.query(question)
        print("\n" + "-"*60)

    # 演示带解释的查询
    print("\n" + "="*60)
    print("演示带解释的查询:")
    print("="*60)

    result = text2sql.query_with_explanation("北京地区的平均保费是多少？")
    print(f"\nSQL:\n{result['sql']}")
    print(f"\n解释:\n{result['explanation']}")

    text2sql.close()


if __name__ == "__main__":
    main()

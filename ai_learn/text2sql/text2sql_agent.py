"""
Text2SQL 基础版本 - 使用 LangChain SQL Agent
这是最简单快速的实现方式
"""

import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Text2SQLAgent:
    """基于 LangChain SQL Agent 的 Text2SQL 实现"""

    def __init__(self, db_path='insurance.db'):
        """
        初始化 Text2SQL Agent

        Args:
            db_path: 数据库文件路径
        """
        # 检查 API Key
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        # 连接数据库
        print("正在连接数据库...")
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        print(f"✓ 数据库连接成功")
        print(f"  包含表: {self.db.get_usable_table_names()}")

        # 初始化 LLM
        print("\n正在初始化 LLM...")
        self.llm = ChatOpenAI(
            model='gpt-4o',  # 或使用 'gpt-4o-mini' 降低成本
            temperature=0
        )
        print(f"✓ LLM 初始化完成")

        # 创建 SQL Agent
        print("\n正在创建 SQL Agent...")
        self.agent = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="zero-shot-react-description",
            verbose=True  # 设置为 True 可以看到推理过程
        )
        print("✓ SQL Agent 创建完成")

    def query(self, question):
        """
        用自然语言查询数据库

        Args:
            question: 自然语言问题

        Returns:
            查询结果
        """
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}\n")

        try:
            result = self.agent.invoke({"input": question})
            return result['output']
        except Exception as e:
            return f"查询出错: {str(e)}"

    def get_table_info(self, table_name=None):
        """
        获取表结构信息

        Args:
            table_name: 表名，如果为 None 则返回所有表
        """
        if table_name:
            return self.db.get_table_info([table_name])
        else:
            tables = self.db.get_usable_table_names()
            info = "数据库包含以下表:\n"
            for table in tables:
                info += f"\n{table}:\n"
                info += self.db.get_table_info([table])
            return info


def main():
    """演示 Text2SQL Agent 的使用"""

    # 创建 Agent
    agent = Text2SQLAgent()

    # 查看表结构
    print("\n" + "="*60)
    print("数据库表结构:")
    print("="*60)
    print(agent.get_table_info())

    # 示例查询
    questions = [
        "有多少个客户？",
        "查询年龄在30到40岁之间的女性客户",
        "统计每个产品的保单数量",
        "查询本月新增的保单",
        "赔付率最高的3个产品是什么？",
        "查询北京地区的平均保费",
    ]

    print("\n" + "="*60)
    print("开始查询示例:")
    print("="*60)

    for question in questions:
        answer = agent.query(question)
        print(f"\n回答:\n{answer}\n")
        print("-"*60)


if __name__ == "__main__":
    main()

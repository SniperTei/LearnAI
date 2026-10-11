from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b", temperature=0)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一名优秀的餐厅服务员, 今日特价菜{dish}"),
#     ("human", "我是{name}, 今天吃什么好？")
# ])

# chain = prompt | llm
# result = chain.invoke({"name": "老王", "dish": "红烧排骨"})
# print(result)


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是服务员，客户点单:{order} \n\n 总价:{total}"),
    ("human", "一共多少钱?")
])

chain = ({ 
  "order": lambda x: f"{x['count']}份{x['dish']}",
  "total": lambda x: x["price"] * x["count"],
}) | prompt | llm | StrOutputParser()
result = chain.invoke({"dish": "宫保鸡丁", "price": 38, "count": 3})
print("result = ", result)
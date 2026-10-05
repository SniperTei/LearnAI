from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

c = [
    ("system", "你是一名资深厨师, 请给{food}提供一份详细的做法和步骤。"),
    ("human", "{user_input}")
]

# template = ChatPromptTemplate(c)
template = ChatPromptTemplate.from_messages(c)

# prompt_value = template.invoke({"food": "宫保鸡丁", "user_input": "请告诉我怎么做宫保鸡丁？"})

# print("第1段吐出:", type(prompt_value))
# print(prompt_value)

llm = ChatOllama(model="qwen3:4b", temperature=0)
# a = llm.invoke(prompt_value)
# print("第2段吐出:", type(a))
# print(a)
# print("====")
parser = StrOutputParser()
# s = parser.invoke(a)
# print("第3段吐出:", type(s))
# print(s)
# print("====")

chain = template | llm | parser

# result = chain.invoke({"food": "宫保鸡丁", "user_input": "请告诉我怎么做宫保鸡丁？"})
results = chain.batch([{"food": "宫保鸡丁", "user_input": "请告诉我怎么做宫保鸡丁？"},
                              {"food": "麻婆豆腐", "user_input": "请告诉我怎么做麻婆豆腐？"},
                              {"food": "糖醋排骨", "user_input": "请告诉我怎么做糖醋排骨？"}])

print(results)
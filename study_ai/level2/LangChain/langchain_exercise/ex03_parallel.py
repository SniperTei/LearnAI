from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
# import time

create_menu_name_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名创意大师，请给{food}提供一个创意的菜名。"),
    ("human", "{user_input}")
])

write_menu_desc_prompt = ChatPromptTemplate.from_messages([
    ("system", "你的文案功底深厚，文学功底扎实，请用100字以内的文字来写一段{food}的描述，要突出特点和卖点。"),
    ("human", "{user_input}")
])

llm = ChatOllama(model="qwen3:4b", temperature=0)

parser = StrOutputParser()

create_menu_name_chain = create_menu_name_prompt | llm | parser
write_menu_desc_chain = write_menu_desc_prompt | llm | parser

# runnableTask = RunnableLambda(lambda inputs: {
#     "menu_name": create_menu_name_chain.invoke(inputs),
#     "menu_desc": write_menu_desc_chain.invoke(inputs)
# })
parallel_runnable = RunnableParallel({
    "menu_name": create_menu_name_chain,
    "menu_desc": write_menu_desc_chain
})

result = parallel_runnable.invoke({"food": "宫保鸡丁", "user_input": "请给我一个创意的菜名和描述。"})

print(result)

# final_chain = {"menu_name": create_menu_name_chain, "menu_desc": write_menu_desc_chain} | RunnableLambda(lambda r: f"菜名：{r['menu_name']}\n介绍：{r['menu_desc']}")
# print(final_chain.invoke({"food": "宫保鸡丁", "user_input": "..."}))

# test = {"a": RunnableLambda(lambda x: f"我收到了:{x}"), "b": RunnableLambda(lambda x: 123)}
# print(test.invoke({"food": "宫保鸡丁"}))

  # 改法1：显式包装（稳妥）
# test = RunnableParallel({
#     "a": RunnableLambda(lambda x: f"我收到了:{x}"),
#     "b": RunnableLambda(lambda x: 123),
# })
# print(test.invoke({"food": "宫保鸡丁"}))

# chain = {"a": RunnableLambda(lambda x: f"我收到了:{x}"), "b": RunnablePassthrough()}
# print((chain | RunnableLambda(lambda r: f"a分支说:{r['a']}, b分支是:{r['b']}")).invoke({"food": "宫保鸡丁"}))
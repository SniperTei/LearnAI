from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough   # 补上
from langchain_community.chat_message_histories import ChatMessageHistory
# version 1

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是餐厅助手。"),
#     ("human", "{question}")
# ])

# llm = ChatOllama(model="qwen3:4b", temperature=0)

# first_chain = prompt | llm | StrOutputParser()
# question = "你好，我想要点一份葱爆羊肉, 一份宫保鸡丁"
# result0 = first_chain.invoke({"question": question})

# print("result0 = ", result0)
# print("-" * 20)
# message_history = ChatMessageHistory()
# message_history.add_user_message(question)
# message_history.add_ai_message(result0)

# prompt_with_history = ChatPromptTemplate.from_messages([
#     ("system", "你是餐厅助手。"),
#     *message_history.messages,
#     ("human", "{question}")
# ])

# second_chain = prompt_with_history | llm | StrOutputParser()
# question2 = "请问我点了几个菜？"
# result1 = second_chain.invoke({"question": question2})
# print("result1 = ", result1)
# print("-" * 20)

# message_history.add_user_message(question2)
# message_history.add_ai_message(result1)
# print("message_history.messages = ", message_history.messages)
# print("-" * 20)

# version 2
# 核心认知：模型本身是金鱼（每次都是全新对话，啥也不记得）
# 所谓"记忆"= 每次提问前，把历史聊天记录整本重发一遍
# v1 是我们手动记账+手动塞；v2 让框架雇一个"自动记账员"替我们干这三件事

from langchain_core.runnables.history import RunnableWithMessageHistory

base_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是餐厅助手。"),
    MessagesPlaceholder(variable_name="history"),   # ← 在模板上挖个洞，名叫"history"，旧账本将从这里塞进去
    ("human", "{question}")                          # ← 本次新问题照常走 {question} 这个洞
])

llm = ChatOllama(model="qwen3:4b", temperature=0)

# ===== 记忆三件套之一：仓库 =====
# 字典当仓库：key=session_id（客人编号），value=该客人的账本(ChatMessageHistory)
# 模块级变量，程序活着它就活着 → 账本不会丢
# 如果每次调用都新建（金鱼lambda干的事），账本用完即焚 → 失忆
history = {}

# ===== 记忆三件套之二：取账员 =====
# 框架每次 invoke 前会打电话问它："session_id=X 的账本拿来"
# 有 → 返回旧账本（接着记）；没有 → 开新账本存进仓库再返回
# 注意坑：必须用 in 判断，写 history[session_id] 第一次访问直接 KeyError
def get_session_history(session_id: str):
    if session_id in history:
        return history[session_id]
    else:
        history[session_id] = ChatMessageHistory()
        return history[session_id]

# ===== 记忆三件套之三：自动记账员上岗 =====
base_chain = RunnableWithMessageHistory(
    base_prompt | llm | StrOutputParser(),   # 原来的链原封不动，只是外包了一层记账服务
    get_session_history=get_session_history, # ← 每次invoke前找取账员拿账本
    input_messages_key="question",           # ← 问框架：输入dict里"question"的值要记为human消息
    history_messages_key="history"           # ← 问框架：旧账本塞进名叫"history"的那个洞
)
# 它在幕后每次 invoke 自动做的事（= v1 里我们手写的）：
#   1. 拿账本：get_session_history(session_id)
#   2. 塞账本：账本messages → MessagesPlaceholder("history") 的洞
#   3. 跑链条：prompt(带历史) | llm | parser
#   4. 记新账：本次question→add_user_message，本次回答→add_ai_message

# session_id 通过 config 传，不占模板的洞 —— 同一条链可以服务无数客人，互不串门
# result0 = base_chain.invoke({"question": "你好，我想要点一份葱爆羊肉, 一份宫保鸡丁"}, config={"configurable": {"session_id": "what"}})
# print("result0 = ", result0)
# print("-" * 50)
# result1 = base_chain.invoke({"question": "请问我点了几个菜？"}, config={"configurable": {"session_id": "what"}})
# print("result1 = ", result1)

result0 = base_chain.invoke({"question": "你好，我想要点一份葱爆羊肉, 一份宫保鸡丁"}, config={"configurable": {"session_id": "userA"}})
print("result0 = ", result0)
print("-" * 50)
result1 = base_chain.invoke({"question": "请问我点了几个菜？"}, config={"configurable": {"session_id": "userB"}})
print("result1 = ", result1)

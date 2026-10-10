from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough   # 补上
from langchain_classic.document_loaders import TextLoader  
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel  

class Menu(BaseModel):
    menu_name: str
    menu_description: str
    ingredients: str
    price: str
    star: int

class MenuItems(BaseModel):
    menus: list[Menu]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是点餐助手，只根据下面的已知信息回答问题，已知信息里没有的内容，就直说无法回答。\n\n已知信息:\n{context}"),
    ("human", "{question}")
])

docs = TextLoader("08menu.md", encoding="utf-8").load()

chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40).split_documents(docs)

embedding_model = OllamaEmbeddings(model="bge-m3")

vector_store = Chroma.from_documents(chunks, embedding_model)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="qwen3:4b", temperature=0)
structure_llm = llm.with_structured_output(MenuItems)

format_docs = lambda docs: "\n\n".join([d.page_content for d in docs])

# chain = (
#     {
#         "context": retriever | format_docs,   # ← 在这！检索器在这里上岗
#         "question": RunnablePassthrough()
#     }
#     | prompt | structure_llm
# )

# result = chain.invoke("最贵的凉菜？")
# print(len(docs))        # 2（或更少，如果库里总共不够2块）
# print(type(docs[0]))    # Document —— 每个元素是一块
# print(result)

# print(type(result))                    # MenuItems，不是 str 不是 dict                       
# print(result.menus[0].menu_name)       # 点号取值，第一道分类名  
# print("total stars : ", sum(m.star for m in result.menus))

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是美食家，只根据下面的已知信息回答问题，已知信息里没有的内容，就直说无法回答。\n\n已知信息:\n{context}, \n\n客人的口味: {flavor}"),
    ("human", "{question}")
])

# chain = ({
#     "context": (lambda q: "客户喜欢{customer_flavor}的") | retriever | format_docs,   # ← 病根1
#     "flavor": lambda x: "清淡",
#     "question": RunnablePassthrough()                                                 # ← 病根2
# } | prompt | llm | StrOutputParser())
#
# print(chain.input_schema.model_json_schema())
# print(chain.invoke({"input": "请推荐一款客人喜欢的菜，价格要贵", "customer_flavor": "荤的凉菜"}))

# ===== 修复版 =====
# 病根1：普通字符串里的 {xxx} 不是洞！是字面花括号。
#        洞只长在 ChatPromptTemplate 身上；lambda 里要填空必须用 f-string（f 前缀）。
#        症状：检索员拿到的采购单是字面文字 "客户喜欢{customer_flavor}的"，
#              "荤""凉菜"等关键词从没到过检索员手里 → 酱牛肉那行根本没被捞回来。
# 病根2：invoke 传 dict 时，分拣台把【整个dict】复制给每条路。
#        RunnablePassthrough 会把整个dict透传进 {question} 洞（被字符串化，脏）。
#        修法：每条路自己用 d["key"] 抠自己要的字段。
chain = ({
    # context路：第一工人是lambda，收整个dict(d)，用f-string缝出正经采购单再交检索员
    "context": (lambda d: f"客户喜欢{d['customer_flavor']}的菜，价格要贵") | retriever | format_docs,
    # flavor路：不再写死"清淡"，从dict里抠口味字段 —— 口径统一，不再打架
    "flavor": lambda d: d["customer_flavor"],
    # question路：不再透传整个dict，只抠问题字段
    "question": lambda d: d["input"],
} | prompt | structure_llm)

print(chain.input_schema.model_json_schema())
print(chain.invoke({"input": "请推荐一款客人喜欢的菜，价格要贵", "customer_flavor": "荤的凉菜"}))
from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.runnables import RunnablePassthrough   # 补上
# from langchain_classic.document_loaders import TextLoader  
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
    ("system", "你是点餐助手"),
    ("human", "{question}")
])

# docs = TextLoader("04menu.txt", encoding="utf-8").load()

# chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40).split_documents(docs)

# embedding_model = OllamaEmbeddings(model="bge-m3")

llm = ChatOllama(model="qwen3:4b", temperature=0)
structure_llm = llm.with_structured_output(MenuItems)

chain = prompt | structure_llm

result = chain.invoke({"question": "我想看一下菜单"})
print(result)

print(type(result))                    # MenuItems，不是 str 不是 dict                       
print(result.menus[0].menu_name)       # 点号取值，第一道分类名  
print("total stars : ", sum(m.star for m in result.menus))


# RAG (Retrieval-Augmented Generation) Example

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough   # 补上

llm = ChatOllama(model="qwen3:4b", temperature=0)
embedding_model = OllamaEmbeddings(model="bge-m3")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是餐厅助手。只根据下面的已知信息回答问题，已知信息里没有的内容，就直说无法回答。\n\n已知信息:\n{context}"),
    ("human", "{question}")
])

docs = TextLoader("04menu.txt", encoding="utf-8").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
chunks = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)

retriever = vector_store.as_retriever()

format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)   # Document列表→字符串

rag_chain = (
    {
        "context": retriever | format_docs,    # 检索→格式化，自动填context
        "question": RunnablePassthrough()       # 问题原样透传
    }
    | prompt | llm | StrOutputParser()
)


# result = rag_chain.invoke("请为我推荐减脂菜")
# print(result)
print("-" * 50)
print(rag_chain.invoke("哪道菜最清淡？"))
print("-" * 50)
print(rag_chain.invoke("菜单里有披萨吗？"))
print("-" * 50)
print(rag_chain.invoke("请为我推荐减脂菜"))
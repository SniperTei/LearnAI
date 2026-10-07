from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma    # 注意：这次用 langchain_chroma（专属对接包）
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = OllamaEmbeddings(model="bge-m3")

# llm = ChatOllama(model="qwen3:4b")

vec = embedding_model.embed_query("红烧排骨真好吃")
print(type(vec))          # 列表
print(len(vec))           # ← 维度！记住这个数
print(vec[:5])            # 前5个数字，感受一下"神秘的浮点数"

docs = TextLoader("04menu.txt", encoding="utf-8").load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
chunks = text_spliter.split_documents(docs)

vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)

print(vector_store.similarity_search("哪道菜最清淡？", k=2))

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
docs2 = retriever.invoke("哪道菜最清淡？")
print(docs2)             

print(vector_store.similarity_search("我在减脂，想吃不长胖的", k=2))               # 还是Document对象！和②结果对比
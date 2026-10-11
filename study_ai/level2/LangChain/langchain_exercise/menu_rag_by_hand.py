from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel  

llm = ChatOllama(model="qwen3:4b", temperature=0)

# 检索那条老手艺全套：TextLoader 读 08menu.md（记得 encoding）→ 切块 → 嵌入 → Chroma → retriever → format_docs

# 读
docs = TextLoader("08menu.md", encoding="utf-8").load()

# 切块, 每个chunk_size=300， chunk_overlap=40
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
chunks = text_splitter.split_documents(docs)

# 嵌入
embedding_model = OllamaEmbeddings(model="bge-m3")

# Chroma
vector_store = Chroma.from_documents(chunks, embedding=embedding_model)

# retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# format_docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 输出模型
class MenuItem(BaseModel):
    name: str # 菜名
    description: str # 菜品描述
    price: float # 价格
    ingredients: list[str] # 食材
    rating: float # 星级

class MenuOutput(BaseModel):
    items: list[MenuItem]

# 结构化LLM
structured_llm = llm.with_structured_output(MenuOutput)

# 模板三洞：{context}（已知信息）、{taste}（客人口味，单独一行，不许贴着已知信息）、{question}
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "只根据已知信息:{context}，没有直说无法回答、\n\n 客人口味:{taste}"),
    ("human", "{question}")
])

chain = ({
    "context": (lambda x: f"{x['question']}，口味{x['taste']}") | retriever | format_docs,
    "taste": lambda x: x["taste"],
    "question": lambda x: x["question"]
}) | prompt_template | structured_llm

# result = chain.invoke({"question": "有什么推荐的", "taste": "清淡"})
# print("result = ", result)
result = chain.invoke({"question": "有什么推荐的", "taste": "荤的凉菜"})
print("result = ", result)
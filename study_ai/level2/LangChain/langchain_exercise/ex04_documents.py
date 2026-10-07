from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


docs = TextLoader("04menu.txt", encoding="utf-8").load()
print(f"type(docs): {type(docs)}")
print(f"type(docs[0]): {type(docs[0])}")

print(f"正文前50字: {docs[0].page_content[:50]}")
print(f"metadata: {docs[0].metadata}")     # ← 重点看：里面记了什么？

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=40
# )

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunks = splitter.split_documents(docs)

print(f"Number of chunks: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print(f"Chunk {index}: {chunk.page_content}")
    print(f"Chunk {index} 长度={len(chunk.page_content)}  metadata={chunk.metadata}")
    print("---------")
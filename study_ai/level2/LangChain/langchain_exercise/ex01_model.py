from langchain_ollama import ChatOllama
import asyncio

llm = ChatOllama(model="qwen3:4b", temperature=0)

messages = [
  ("system", "You are a helpful assistant that translates English to Korean. Translate the user sentence."),
  ("human", "I love programming")
]

# ai_msg = llm.invoke(messages)
# # ai_msg
# print(ai_msg.content)

async def translate_message(message):
    resp = await llm.ainvoke(message)
    # print(resp)
    return resp

ai_msg2 = asyncio.run(translate_message("Today is a sunny day."))

print(ai_msg2.content)

print("-" * 20)

print(ai_msg2)

print("----- end -----")
import ollama

prompt = "请介绍一下大模型蒸馏技术？"
response = ollama.generate(model="deepseek-r1:1.5b", prompt=prompt).response
print(response)
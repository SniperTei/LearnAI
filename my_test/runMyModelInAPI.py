import requests
import json

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-r1:1.5b",
    "prompt": "are you ok?",
    "stream": False
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
# 从result中提取response字段
response_text = result["response"]
# 打印response字段
print(response_text)

# print(json.dumps(result, indent=2))

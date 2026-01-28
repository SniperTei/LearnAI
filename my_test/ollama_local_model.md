# 本地大模型 API 调用指南

本文档介绍如何在本地通过 API 调用大语言模型，以 Ollama 为例。

## 目录

- [环境准备](#环境准备)
- [安装 Ollama](#安装-ollama)
- [下载模型](#下载模型)
- [启动服务](#启动服务)
- [Python 调用示例](#python-调用示例)
- [API 参数说明](#api-参数说明)
- [常见问题](#常见问题)

## 环境准备

### 系统要求

- macOS / Linux / Windows
- Python 3.7+
- 至少 8GB 内存（推荐 16GB+）
- 足够的磁盘空间（模型文件通常几GB）

### Python 依赖

```bash
pip install requests
```

## 安装 Ollama

### macOS

```bash
brew install ollama
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

下载安装包：https://ollama.com/download

## 下载模型

### 查看可用模型

访问 https://ollama.com/library 查看所有可用模型。

### 下载模型示例

```bash
# 下载 deepseek-r1 1.5b 版本
ollama pull deepseek-r1:1.5b

# 下载 llama3.2 3b 版本
ollama pull llama3.2:3b

# 下载 qwen2.5 7b 版本
ollama pull qwen2.5:7b
```

### 查看已安装模型

```bash
ollama list
```

## 启动服务

### 启动 Ollama 服务

```bash
ollama serve
```

服务默认运行在 `http://localhost:11434`

### 验证服务状态

```bash
curl http://localhost:11434/api/tags
```

## Python 调用示例

### 基础调用

```python
import requests
import json

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-r1:1.5b",
    "prompt": "你好，请介绍一下你自己",
    "stream": False
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

# 提取响应文本
response_text = result["response"]
print(response_text)

# 打印完整响应（调试用）
print(json.dumps(result, indent=2))
```

### 流式输出调用

```python
import requests

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-r1:1.5b",
    "prompt": "写一首关于春天的诗",
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)

for line in response.iter_lines():
    if line:
        result = json.loads(line)
        if "response" in result:
            print(result["response"], end="", flush=True)
```

### 带参数的调用

```python
import requests
import json

url = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-r1:1.5b",
    "prompt": "解释什么是机器学习",
    "stream": False,
    "options": {
        "temperature": 0.7,      # 控制随机性，0-2，越高越随机
        "top_p": 0.9,            # 核采样参数，0-1
        "max_tokens": 500,       # 最大生成 token 数
        "num_ctx": 2048          # 上下文窗口大小
    }
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result["response"])
```

### 对话模式调用

```python
import requests
import json

url = "http://localhost:11434/api/chat"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "deepseek-r1:1.5b",
    "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
        {"role": "user", "content": "请介绍一下Python"}
    ],
    "stream": False
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print(result["message"]["content"])
```

## API 参数说明

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 是 | 模型名称，如 "deepseek-r1:1.5b" |
| prompt | string | 是* | 输入提示词（generate API） |
| messages | array | 是* | 对话消息列表（chat API） |
| stream | boolean | 否 | 是否流式输出，默认 false |
| options | object | 否 | 模型参数配置 |
| format | string | 否 | 输出格式，如 "json" |

### options 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| temperature | float | 0.8 | 控制随机性，0-2 |
| top_p | float | 0.9 | 核采样参数，0-1 |
| top_k | int | 40 | 采样候选数 |
| num_predict | int | -1 | 最大生成 token 数 |
| num_ctx | int | 2048 | 上下文窗口大小 |
| repeat_penalty | float | 1.1 | 重复惩罚系数 |

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| model | string | 使用的模型名称 |
| response | string | 生成的文本 |
| done | boolean | 是否完成 |
| context | array | 上下文 token 列表 |
| total_duration | int | 总耗时（纳秒） |
| load_duration | int | 加载耗时（纳秒） |
| prompt_eval_count | int | 提示词 token 数 |
| eval_count | int | 生成 token 数 |

## 常见问题

### 1. 连接失败

**问题**：`Connection refused`

**解决方案**：
- 确认 Ollama 服务已启动：`ollama serve`
- 检查端口是否正确：默认 11434
- 检查防火墙设置

### 2. 模型不存在

**问题**：`model 'xxx' not found`

**解决方案**：
- 使用 `ollama list` 查看已安装模型
- 使用 `ollama pull <model_name>` 下载模型

### 3. 内存不足

**问题**：运行缓慢或崩溃

**解决方案**：
- 选择更小的模型（如 1.5b、3b）
- 减小 `num_ctx` 参数
- 关闭其他占用内存的程序

### 4. 响应速度慢

**优化建议**：
- 使用量化模型（如 :q4_0 后缀）
- 减小 `num_ctx` 参数
- 使用更小的模型
- 考虑使用 GPU 加速（需要支持 CUDA 的显卡）

### 5. 中文支持

**问题**：中文输出质量差

**解决方案**：
- 选择支持中文的模型（如 qwen、deepseek-r1）
- 在 prompt 中明确要求使用中文
- 调整 temperature 参数

## 进阶用法

### 自定义模型

```bash
# 创建 Modelfile
cat > Modelfile <<EOF
FROM deepseek-r1:1.5b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM 你是一个专业的Python编程助手。
EOF

# 创建并运行模型
ollama create my-python-assistant -f Modelfile
ollama run my-python-assistant
```

### 批量处理

```python
import requests
import json

def generate_response(prompt, model="deepseek-r1:1.5b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]

prompts = ["你好", "介绍一下Python", "什么是机器学习"]
for prompt in prompts:
    print(f"问题: {prompt}")
    print(f"回答: {generate_response(prompt)}\n")
```

## 参考资源

- Ollama 官网：https://ollama.com
- Ollama API 文档：https://github.com/ollama/ollama/blob/main/docs/api.md
- 模型库：https://ollama.com/library

## 许可证

本文档基于实际使用经验编写，仅供参考。

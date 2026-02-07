# LangChain 天气查询助手

一个使用 LangChain 框架构建的智能天气查询应用，可以通过自然语言查询各地天气信息。

## 功能特性

- ✨ 使用 LangChain Agent 架构
- 🛠️ 集成天气查询工具（支持实时和模拟数据）
- 💬 对话记忆功能
- 🌍 支持中文和英文城市名
- 📅 支持当前天气和天气预报查询

## 项目结构

```
langchain_a/
├── weather_agent.py    # 主程序（Agent 应用）
├── weather_tools.py    # 天气查询工具模块
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量示例
└── README.md          # 说明文档
```

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 OpenAI API Key：

```
OPENAI_API_KEY=sk-your-api-key-here
```

> 获取 API Key: https://platform.openai.com/api-keys

## 使用方法

### 交互式使用

直接运行主程序：

```bash
python weather_agent.py
```

然后可以提问：
- "北京今天天气怎么样？"
- "上海明天会下雨吗？"
- "查询广州未来3天天气预报"
- "New York 的天气"

### 代码调用

```python
from weather_agent import simple_query

# 查询天气
result = simple_query("北京今天天气怎么样？")
print(result)
```

## 技术架构

### LangChain 组件

1. **LLM (ChatOpenAI)**: 使用 GPT-3.5-turbo 作为语言模型
2. **Tools**: 自定义天气查询工具
   - `get_weather`: 查询当前天气
   - `get_forecast`: 获取天气预报
3. **Agent**: 使用 Tool Calling Agent 模式
4. **Memory**: ConversationBufferMemory 保存对话历史
5. **Prompt Template**: 结构化的提示词模板

### 工作流程

```
用户输入
    ↓
Agent 解析意图
    ↓
选择合适的工具
    ↓
执行工具获取数据
    ↓
LLM 生成自然语言回复
    ↓
返回给用户
```

## 核心代码说明

### 1. 工具定义 (weather_tools.py)

使用 `@tool` 装饰器定义 LangChain 工具：

```python
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息"""
    # 实现逻辑
```

### 2. Agent 创建 (weather_agent.py)

```python
# 创建 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([...])

# 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)
```

## 依赖说明

- `langchain`: LangChain 核心框架
- `langchain-openai`: OpenAI 模型集成
- `langchain-community`: 社区扩展
- `python-dotenv`: 环境变量管理
- `requests`: HTTP 请求（调用天气 API）

## API 说明

本项目使用免费的 wttr.in 天气 API：
- 无需 API Key
- 支持全球城市
- 提供实时和预报数据

如需更好的数据质量，可以集成其他服务（如 OpenWeatherMap）。

## 扩展建议

1. **添加更多工具**: 如空气质量、紫外线指数等
2. **支持语音输入输出**: 集成语音识别和 TTS
3. **多语言支持**: 扩展更多语言
4. **数据库集成**: 保存查询历史
5. **Web 界面**: 使用 Streamlit 或 Gradio 构建 UI

## 常见问题

**Q: 提示 API Key 错误？**
A: 检查 `.env` 文件是否正确配置，或 API Key 是否有效。

**Q: 天气数据不准确？**
A: 默认使用免费 API，可以替换为更专业的天气服务。

**Q: 如何使用其他 LLM？**
A: 修改 `ChatOpenAI` 为其他提供商，如 `ChatAnthropic`。

## License

MIT

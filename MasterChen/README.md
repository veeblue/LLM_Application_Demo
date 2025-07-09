# 陈半仙·AI 算命先生

🧙‍♂️ 一个基于 LangChain、FastAPI、Gradio 的 AI 算命先生，支持八字排盘、紫薇斗数、姓名测算、运势查询等功能，具备情绪识别与多工具调用能力。

## 主要功能
- 聊天式算命（八字、姓名、紫薇斗数等）
- 支持情绪识别，自动调整回复风格
- 可查询本地知识库（如生肖运势）
- 支持实时搜索工具
- 八字排盘自动参数提取与外部API调用
- Web API 与 Gradio 可视化界面

## 依赖环境
- Python 3.8+
- fastapi
- uvicorn
- gradio
- langchain
- langchain_community
- langchain_huggingface
- langchain_openai
- python-dotenv
- requests
- chromadb

> 建议使用 `pip install` 安装上述依赖。可根据实际代码补充 requirements.txt。

## 安装与运行
1. **克隆项目**
   ```bash
   git clone <repo_url>
   cd MasterChen
   ```
2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   # 或手动安装依赖
   pip install fastapi uvicorn gradio langchain langchain_community langchain_huggingface langchain_openai python-dotenv requests chromadb
   ```
3. **配置环境变量**
   在根目录下创建 `.env` 文件，内容示例：
   ```env
   CHAT_MODEL=deepseek-chat
   DEEPSEEK_API_KEY=你的DeepSeek密钥
   DEEPSEEK_BASE_URL=你的DeepSeek API地址
   DASHSCOPE_API_KEY=你的DashScope密钥
   DASHSCOPE_BASE_URL=你的DashScope API地址
   YUANFENJU_API_KEY=你的缘分居API密钥
   SERPAPI_API_KEY=你的SerpAPI密钥
   ```

4. **启动服务**
   ```bash
   python server.py
   ```
   默认会通过 Gradio 启动 Web 界面，访问 http://localhost:7860

## API 说明
- `GET /`  健康检查
- `POST /chat`  聊天接口，参数：`query`（问题），`session_id`（可选）
- `POST /add_urls`  添加URL知识库（预留）
- `POST /add_pdfs`  添加PDF知识库（预留）
- `POST /add_texts`  添加文本知识库（预留）
- `WebSocket /ws`  实时通信接口

## Gradio 可视化界面
- 支持会话ID、头像、气泡式对话
- 输入问题后点击“发送”或回车即可体验
- 仅供娱乐，结果不作为任何决策依据

## 本地知识库与八字API
- 本地知识库基于 ChromaDB，支持生肖运势等查询
- 八字排盘调用 [缘分居API](https://api.yuanfenju.com/)

## 免责声明
本项目仅供娱乐，所有算命结果不具备科学依据，请勿用于实际决策。


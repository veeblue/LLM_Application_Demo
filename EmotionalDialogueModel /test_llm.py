from langchain_openai import ChatOpenAI
import os
# 初始化模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model_name="qwen-plus",
    temperature=0.8
)

print(llm.invoke("你好"))
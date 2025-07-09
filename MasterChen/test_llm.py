from langchain_openai import ChatOpenAI
import os


llm = ChatOpenAI(
            model='qwen-plus',  # 或者 'qwen-turbo', 'qwen-max' 等
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base=os.getenv("DASHSCOPE_BASE_URL"),
            temperature=0,
        )

print(llm.invoke("你好"))
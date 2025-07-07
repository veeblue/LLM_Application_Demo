    
from llama_index.llms.openai_like import OpenAILike
from config import Config
import os

print(os.getenv("DEEPSEEK_API_KEY"))
print(os.getenv("DEEPSEEK_BASE_URL"))
llm = OpenAILike(
    model="qwen-plus",
    api_key=Config.DASHSCOPE_API_KEY,
    api_base=Config.DASHSCOPE_BASE_URL,
    is_chat_model=True,
)
# llm = OpenAILike(
# model="deepseek-r1:1.5b",
# api_base="http://localhost:11434/v1",
# api_key="fake",
# context_window=4096,
# is_chat_model=True,
# is_function_calling_model=False,
# )
res = llm.complete("你好")
print(res)
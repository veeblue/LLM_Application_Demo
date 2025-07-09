import os
from dotenv import load_dotenv

# 自动加载 .env 文件
load_dotenv()

class Config:
    CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_BASE_URL")
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DASHSCOPE_API_BASE = os.getenv("DASHSCOPE_BASE_URL")
    YUANFENJU_API_KEY = os.getenv("YUANFENJU_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    

# 用法示例：
# from config import Config
# api_key = Config.DEEPSEEK_API_KEY 
import os

class Config:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

    CHROMA_DB_PATH = "chroma_db"
    DATA_DIR = "./data/"
    PERSIST_DIR = "./storage"
    VECTOR_DB_DIR = "./chroma_db"

    COLLECTION_NAME = "chinese_labor_laws"
    TOP_K = 10  # 扩大初始检索数量
    RERANK_TOP_K = 3  # 重排序后保留数量

    RERANK_MODEL = "./models/bge-reranker-large"
    EMBED_MODEL = "./models/fixed-text2vec-base-chinese-sentence"
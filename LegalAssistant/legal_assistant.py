import json
from typing import List, Dict

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank  # 新增重排序组件
from config import Config
from pathlib import Path


# ================== 加载数据 ==================
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"

    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 验证数据结构
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                all_data.extend({
                                    "content": item,
                                    "metadata": {"source": json_file.name}
                                } for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")

    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data

# ================== 创建节点 ==================
def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """添加ID稳定性保障"""
    nodes = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]

        for full_title, content in law_dict.items():
            # 生成稳定ID（避免重复）
            node_id = f"{source_file}::{full_title}"

            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"

            node = TextNode(
                text=content,
                id_=node_id,  # 显式设置稳定ID
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)

    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes
# ================== 初始化模型 ==================
def init_models():
    """初始化LLM"""
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

    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL,
        device="mps"
    )

    # 重排序器
    reranker = SentenceTransformerRerank(
        model=Config.RERANK_MODEL,
        top_n=Config.RERANK_TOP_K
    )

    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.reranker = reranker

    test_embedding = embed_model.get_text_embedding("中华人民共和国劳动法")
    print(f"Embedding维度验证：{len(test_embedding)}")


    return llm, embed_model, reranker

# ================== 初始化向量存储 ==================
def init_vector_store(nodes: List[TextNode])-> VectorStoreIndex:
    """初始化向量存储"""
    chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
    # 创建或获取ChromaDB集合，用于存储法律条款的向量表示
    # Config.COLLECTION_NAME: 集合名称，在config.py中定义为"chinese_labor_laws"
    # metadata={"hnsw:space": "cosine"}: 设置向量相似度计算方式为余弦相似度
    # 余弦相似度适合用于文本语义相似度计算，值域为[-1,1]，1表示完全相同
    chroma_collection = chroma_client.get_or_create_collection(
        Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"})

    # 创建存储上下文，使用ChromaDB作为向量存储
    # ChromaVectorStore将ChromaDB集合包装为LlamaIndex可用的向量存储接口
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    if chroma_collection.count() == 0 and nodes is not None:
        print(f"创建新索引（{len(nodes)}个节点）...")

        # 显式将节点添加到存储上下文
        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        # 双重持久化保障
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)  # <-- 新增
    else:
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # 安全验证
    print("\n存储验证结果：")
    doc_count = len(storage_context.docstore.docs)
    print(f"DocStore记录数：{doc_count}")

    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"示例节点ID：{sample_key}")
    else:
        print("警告：文档存储为空，请检查节点添加逻辑！")

    return index





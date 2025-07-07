import os
from pathlib import Path
import time
from typing import List

import streamlit as st

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.utils import get_response_synthesizer
import gradio as gr
from llama_index.core.schema import TextNode

from config import Config
from legal_assistant import load_and_validate_json_files, create_nodes, init_models, init_vector_store

# ================== Streamlit页面配置 ==================
st.set_page_config(
    page_title="智能劳动法咨询助手",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

# ================== 禁用Streamlit文件监控 ==================
# def disable_streamlit_watcher():
#     """Patch Streamlit to disable file watcher"""
#
#     def _on_script_changed(_):
#         return
#
#     from streamlit import runtime
#     runtime.get_instance()._on_script_changed = _on_script_changed

@st.cache_resource(show_spinner="正在加载模型...")
def st_init_models():
    """初始化模型"""
    embed_model, llm, reranker = init_models()
    return embed_model, llm, reranker

@st.cache_resource(show_spinner="正在加载数据...")
def st_init_vector_store(_nodes):
    return init_vector_store(_nodes)

def init_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("cleaned", msg["content"])  # 优先使用清理后的内容

        with st.chat_message(role):
            st.markdown(content)

            # 如果是助手消息且包含思维链
            if role == "assistant" and msg.get("think"):
                with st.expander("📝 模型思考过程（历史对话）"):
                    for think_content in msg["think"]:
                        st.markdown(f'<span style="color: #808080">{think_content.strip()}</span>',
                                    unsafe_allow_html=True)

            # 如果是助手消息且有参考依据（需要保持原有参考依据逻辑）
            if role == "assistant" and "reference_nodes" in msg:
                show_reference_details(msg["reference_nodes"])


def show_reference_details(nodes):
    with st.expander("查看支持依据"):
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            st.markdown(f"**[{idx}] {meta['full_title']}**")
            st.caption(f"来源文件：{meta['source_file']} | 法律名称：{meta['law_name']}")
            st.markdown(f"相关度：`{node.score:.4f}`")
            # st.info(f"{node.node.text[:300]}...")
            st.info(f"{node.node.text}")

import re
def main():
    # 禁用 Streamlit 文件热重载
    # disable_streamlit_watcher()
    st.title("⚖️ 智能劳动法咨询助手")
    st.markdown("欢迎使用劳动法智能咨询系统，请输入您的问题，我们将基于最新劳动法律法规为您解答。")

    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []

    # 加载模型和索引
    embed_model, llm, reranker = st_init_models()

    # 初始化数据
    if not Path(Config.VECTOR_DB_DIR).exists():
        with st.spinner("正在构建知识库..."):
            raw_data = load_and_validate_json_files(Config.DATA_DIR)
            nodes = create_nodes(raw_data)
    else:
        nodes = None

    index = st_init_vector_store(nodes)
    retriever = index.as_retriever(similarity_top_k=Config.TOP_K, vector_store_query_mode="hybrid", alpha=0.5)
    from llama_index.core.response_synthesizers import TreeSummarize
    from llama_index.llms.huggingface import HuggingFaceLLM
    response_synthesizer = TreeSummarize(llm=embed_model)
    # 聊天界面
    init_chat_interface()

    if prompt := st.chat_input("请输入劳动法相关问题"):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 处理查询
        with st.spinner("正在分析问题..."):
            start_time = time.time()

            # 检索流程
            initial_nodes = retriever.retrieve(prompt)
            reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_str=prompt)

            # 过滤节点
            MIN_RERANK_SCORE = 0.4
            filtered_nodes = [node for node in reranked_nodes if node.score > MIN_RERANK_SCORE]

            if not filtered_nodes:
                response_text = "⚠️ 未找到相关法律条文，请尝试调整问题描述或咨询专业律师。"
            else:
                # 生成回答
                response = response_synthesizer.synthesize(prompt, nodes=filtered_nodes)
                response_text = response.response

            # 显示回答
            with st.chat_message("assistant"):
                # 提取思维链内容并清理响应文本
                think_contents = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
                cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

                # 显示清理后的回答
                st.markdown(cleaned_response)

                # 如果有思维链内容则显示
                if think_contents:
                    with st.expander("📝 模型思考过程（点击展开）"):
                        for content in think_contents:
                            st.markdown(f'<span style="color: #808080">{content.strip()}</span>',
                                        unsafe_allow_html=True)

                # 显示参考依据（保持原有逻辑）
                show_reference_details(filtered_nodes[:3])

            # 添加助手消息到历史（需要存储原始响应）
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,  # 保留原始响应
                "cleaned": cleaned_response,  # 存储清理后的文本
                "think": think_contents  # 存储思维链内容
            })

if __name__ == "__main__":
    main()
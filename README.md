# LLM_Application_Demo

本仓库包含了我在学习和实践大语言模型（LLM）相关技术过程中开发的项目示例，包括多风格情感对话生成系统、智能法律助手等，涵盖数据处理、模型微调、本地推理部署与多端交互等完整流程。

---

## 目录

- [EmotionalDialogueModel](#emotionaldialoguemodel)
- [LegalAssistant](#legalassistant)

---

## EmotionalDialogueModel

> 基于 LlamaFactory 微调与 vLLM 本地部署的多风格情感对话生成系统

### 项目简介
支持多种情绪风格的对话生成，涵盖数据自动生成、格式转换、模型微调、推理部署和 Web 交互等完整流程。适用于情感陪伴、虚拟助手等场景。

### 功能特性
- 多情绪风格对话模板设计与自动生成
- 数据格式自动转换，兼容 LlamaFactory 训练标准
- 支持单轮/多轮对话数据训练
- 基于 LlamaFactory 的本地微调
- vLLM 本地高效推理部署
- Gradio Web UI 实时交互与参数调优

### 技术栈
Python、LlamaFactory、vLLM、Gradio、数据处理

### 快速开始
```bash
# 进入 EmotionalDialogueModel 目录
cd EmotionalDialogueModel

# 生成多风格对话数据
python generate_data.py

# 转换为 LlamaFactory 格式
python data_convert.py

# 按 LlamaFactory 官方文档微调模型
# 部署 vLLM 并启动 Web UI
python webui.py
```

### 参考资料
- [情绪对话模型微调实录](https://blog.veeblue.com/2025/07/07/%E8%AE%B0%EF%BC%9A%E5%BE%AE%E8%B0%83%E4%B8%80%E4%B8%AA%E6%83%85%E7%BB%AA%E5%AF%B9%E8%AF%9D%E6%A8%A1%E5%9E%8B/)

---

## LegalAssistant

> 基于 LlamaIndex 和 RAG 技术的智能法律问答系统

### 项目简介
专注于中华人民共和国劳动法及劳动合同法的智能法律问答系统，支持批量导入法律条款，集成多种大语言模型，提供高效、准确的法律咨询服务。

### 功能特性
- 法律条款自动抓取、解析与批量导入
- 中文嵌入模型+BGE重排序的高精度法律条款检索
- ChromaDB 持久化向量存储
- LlamaIndex 框架下的 RAG 智能问答
- Streamlit/Gradio/命令行多端交互
- 支持多种主流大语言模型灵活切换

### 技术栈
Python、LlamaIndex、ChromaDB、HuggingFace、BGE Reranker、Streamlit、Gradio、RAG

### 快速开始
```bash
# 进入 LegalAssistant 目录
cd LegalAssistant

# 抓取并生成法律条款数据
python get_legal_clauses.py

# 启动 Streamlit Web 界面
streamlit run app.py

# 或 Gradio/命令行
python legal_assistant.py
```

---

## 亮点
- 多情绪风格与法律场景的 LLM 应用全流程实践
- 支持本地微调、本地推理与多端交互
- 数据、配置、模型流程高度自动化与规范化

---



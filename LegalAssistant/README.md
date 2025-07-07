# ⚖️ 法律助手 (Legal Assistant)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/) [![macOS](https://img.shields.io/badge/macOS-Supported-green.svg)](https://www.apple.com/macos/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Latest-green.svg)](https://github.com/run-llama/llama_index)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 基于LlamaIndex和RAG技术的智能法律问答系统，专门回答中华人民共和国劳动法和劳动合同法相关问题。

## 🚀 功能特点

- 🤖 **智能问答**: 基于向量检索技术，快速找到相关法律条款
- 📚 **权威数据**: 基于官方发布的劳动法和劳动合同法条款
- 🔍 **精准检索**: 使用中文友好的嵌入模型，提高检索准确性
- 💻 **多界面支持**: 支持Streamlit Web界面、Gradio界面和命令行界面
- 📖 **详细引用**: 提供相关法律条款的完整内容和出处
- 🧠 **重排序优化**: 使用BGE重排序模型提高检索精度
- 💾 **持久化存储**: 支持向量数据库持久化，避免重复构建索引
- 🔧 **灵活配置**: 支持多种LLM模型（通义千问、DeepSeek等）

## 📁 项目结构

```
Legal_Assistant/
├── 📄 app.py                    # Streamlit Web应用
├── 📄 legal_assistant.py        # 法律助手主程序（Gradio + CLI）
├── 📄 get_legal_clauses.py      # 法律条款抓取脚本
├── 📄 config.py                 # 配置文件
├── 📄 test_llm.py              # LLM模型测试脚本
├── 📁 data/
│   └── 📄 all_legal_clauses.json # 法律条款数据
├── 📁 models/                   # 本地模型目录
│   ├── 📁 bge-reranker-large/   # BGE重排序模型
│   └── 📁 fixed-text2vec-base-chinese-sentence/ # 中文嵌入模型
├── 📁 chroma_db/               # ChromaDB向量数据库
├── 📁 storage/                 # LlamaIndex存储目录
└── 📄 README.md                # 项目说明文档
```

## 🛠️ 快速开始

### 1️⃣ 克隆项目

```bash
git clone https://github.com/yourusername/legal-assistant.git
cd legal-assistant
```

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install llama-index openai chromadb sentence-transformers gradio streamlit requests beautifulsoup4
```

### 3️⃣ 环境配置

创建 `.env` 文件并配置环境变量：

```bash
# 通义千问配置
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1

# DeepSeek配置（可选）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=your_deepseek_base_url_here
```

### 4️⃣ 获取法律数据

```bash
python get_legal_clauses.py
```

### 5️⃣ 启动应用

#### 🎨 Streamlit界面（推荐）

```bash
streamlit run app.py
```

#### 🖥️ Gradio界面

```bash
python legal_assistant.py
```

#### 💻 命令行界面

```bash
python legal_assistant.py
```

## 🧪 测试

测试LLM模型连接：

```bash
python test_llm.py
```

## 💡 使用示例

### 示例问题

以下是一些可以尝试的问题示例：

| 问题类型 | 示例问题 |
|---------|---------|
| 试用期 | 试用期最长可以约定多长时间？ |
| 合同解除 | 用人单位可以随意解除劳动合同吗？ |
| 加班费 | 加班费如何计算？ |
| 女职工保护 | 女职工在孕期有什么特殊保护？ |
| 合同条款 | 劳动合同必须具备哪些条款？ |
| 工资支付 | 工资支付有什么规定？ |
| 社会保险 | 社会保险如何缴纳？ |
| 工伤处理 | 工伤事故如何处理？ |


### 核心技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **LlamaIndex** | Latest | RAG框架，提供索引和检索功能 |
| **HuggingFace** | Latest | 中文文本嵌入模型 |
| **BGE Reranker** | Latest | 重排序模型，提高检索精度 |
| **ChromaDB** | Latest | 向量数据库，支持持久化存储 |
| **Streamlit** | Latest | 现代化Web界面框架 |
| **OpenAILike** | Latest | 支持多种LLM模型的统一接口 |

### 配置说明

主要配置项（`config.py`）：

```python
TOP_K = 10              # 初始检索数量
RERANK_TOP_K = 3        # 重排序后保留数量
COLLECTION_NAME = "chinese_labor_laws"  # 向量数据库集合名
CHROMA_DB_PATH = "chroma_db"            # 向量数据库路径
PERSIST_DIR = "./storage"               # LlamaIndex存储目录
RERANK_MODEL = "./models/bge-reranker-large"  # 重排序模型路径
EMBED_MODEL = "./models/text2vec-base-chinese-sentence"  # 嵌入模型路径
```

## 📊 性能优化

- ✅ **向量数据库持久化**: 首次构建索引后，后续启动直接加载
- ✅ **模型缓存**: 使用Streamlit缓存机制，避免重复加载模型
- ✅ **重排序优化**: 使用BGE重排序模型提高检索精度
- ✅ **相似度过滤**: 设置最小相似度阈值，过滤低质量检索结果
- ✅ **ID稳定性**: 使用稳定的节点ID，避免重复索引

## 📚 数据来源

| 法律文件 | 来源网站 | 状态 |
|---------|---------|------|
| 中华人民共和国劳动法 | 应急管理部官网 | ✅ 已抓取 |
| 中华人民共和国劳动合同法 | 国家信访局官网 | ✅ 已抓取 |

## ⚠️ 注意事项

- 🔴 **首次运行**: 首次运行时会下载中文嵌入模型和重排序模型，可能需要一些时间
- 🔴 **网络连接**: 需要网络连接来下载模型和数据
- 🔴 **法律建议**: 本系统仅供参考，具体法律问题请咨询专业律师
- 🔴 **数据更新**: 法律条款数据基于抓取时的版本，如有更新请重新运行抓取脚本
- 🔴 **存储空间**: 模型文件较大，确保有足够的磁盘空间
- 🔴 **API密钥**: 确保正确配置LLM服务的API密钥

## 🔧 故障排除

### 常见问题

<details>
<summary>🤖 模型下载失败</summary>

- 检查网络连接
- 尝试使用VPN或代理
- 手动下载模型到 `models/` 目录
</details>

<details>
<summary>📁 数据文件不存在</summary>

- 确保已运行 `get_legal_clauses.py` 获取数据
</details>

<details>
<summary>📦 依赖安装失败</summary>

- 尝试升级pip: `pip install --upgrade pip`
- 使用conda环境: `conda create -n legal python=3.10`
</details>

<details>
<summary>🌐 Web界面无法访问</summary>

- 检查端口是否被占用
- 尝试修改端口号
</details>

<details>
<summary>💾 向量数据库错误</summary>

- 删除 `chroma_db/` 和 `storage/` 目录重新构建
- 检查磁盘空间是否充足
</details>

<details>
<summary>🔑 LLM连接失败</summary>

- 检查API密钥配置
- 验证网络连接
- 确认服务端点地址正确
</details>


## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LlamaIndex](https://github.com/run-llama/llama_index) - RAG框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [Streamlit](https://streamlit.io/) - Web应用框架
- [HuggingFace](https://huggingface.co/) - 模型库



# RAG-hub 知识库问答系统

基于 RAG（检索增强生成）技术的知识库问答系统，支持文档导入、向量化存储、智能问答等功能。

## 📋 项目简介

RAG-hub 是一个企业级知识库解决方案，采用 LangGraph 编排工作流，支持：

- **文档导入**：PDF/Markdown 文件解析、智能切分、向量化入库
- **智能问答**：基于向量检索 + 知识图谱的混合检索，支持流式输出
- **多模态支持**：集成视觉模型，支持图文理解

## 🏗️ 项目结构

```
rag-hub/
├── app/
│   ├── clients/              # 数据库客户端
│   │   ├── milvus_utils.py   # Milvus 向量数据库
│   │   ├── minio_utils.py    # MinIO 对象存储
│   │   ├── mongo_history_utils.py  # MongoDB 历史记录
│   │   └── neo4j_utils.py    # Neo4j 知识图谱
│   ├── conf/                 # 配置文件
│   ├── core/                 # 核心工具（日志、Prompt加载）
│   ├── import_process/       # 文档导入流程
│   │   ├── agent/            # LangGraph 工作流
│   │   ├── api/              # FastAPI 服务
│   │   └── page/             # 前端页面
│   ├── query_process/        # 查询处理流程
│   │   ├── agent/            # LangGraph 工作流
│   │   ├── api/              # FastAPI 服务
│   │   └── page/             # 前端页面
│   ├── lm/                   # 大模型工具（Embedding、Reranker）
│   ├── utils/                # 工具函数
│   └── tool/                 # 模型下载工具
├── prompts/                  # Prompt 模板文件
├── test/                     # 测试代码
├── .env.example              # 环境变量模板
├── environment.yml           # Conda 环境配置
└── .gitignore                # Git 忽略配置
```

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **Web 框架** | FastAPI + Uvicorn |
| **流程编排** | LangChain + LangGraph |
| **向量数据库** | Milvus |
| **对象存储** | MinIO |
| **图数据库** | Neo4j |
| **文档数据库** | MongoDB |
| **模型框架** | PyTorch + Transformers |
| **模型平台** | ModelScope |
| **Embedding** | BGE-M3 |
| **Reranker** | BGE-Reranker |
| **大模型** | 阿里云百炼（Qwen 系列） |

## 📦 环境要求

- Python 3.11+
- Conda（推荐使用）
- Milvus（向量数据库）
- MinIO（对象存储）
- MongoDB（历史记录存储）
- Neo4j（知识图谱，可选）

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/rag-hub.git
cd rag-hub
```

### 2. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate rag-hub
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的配置
# 必须配置的项目：
# - OPENAI_API_KEY：阿里云百炼 API Key
# - MILVUS_URL：Milvus 连接地址
# - MINIO_ENDPOINT：MinIO 服务地址
# - MONGO_URL：MongoDB 连接地址
```

### 4. 下载模型

项目使用本地模型，首次运行需要下载：

```bash
# 下载 Embedding 模型（BGE-M3）
python -m app.tool.download_bgem3

# 下载 Reranker 模型（可选）
python -m app.tool.download_reranker
```

### 5. 启动服务

#### 启动文件导入服务（端口 8000）

```bash
python -m app.import_process.api.file_import_service
```

访问 http://127.0.0.1:8000/import.html 查看导入页面

#### 启动查询服务（端口 8001）

```bash
python -m app.query_process.api.query_service
```

访问 http://127.0.0.1:8001/chat.html 查看聊天页面

## 📡 API 接口

### 文件导入服务（端口 8000）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传文件，支持批量上传 |
| `/status/{task_id}` | GET | 查询任务处理状态 |
| `/import.html` | GET | 文件导入页面 |

### 查询服务（端口 8001）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/query` | POST | 提交查询请求 |
| `/stream/{session_id}` | GET | SSE 流式获取结果 |
| `/history/{session_id}` | GET | 查询会话历史 |
| `/history/{session_id}` | DELETE | 清空会话历史 |
| `/chat.html` | GET | 聊天页面 |

## ⚙️ 配置说明

详细配置请参考 `.env.example` 文件，主要包含：

- **大模型配置**：API Key、模型选择、温度参数
- **向量数据库**：Milvus 连接地址、集合名称
- **对象存储**：MinIO 端点、访问密钥
- **图数据库**：Neo4j 连接配置
- **文档数据库**：MongoDB 连接配置
- **模型路径**：本地模型存放路径
- **日志配置**：日志级别、保留天数

## 📝 工作流程

### 文档导入流程

```
上传文件 → PDF/MD解析 → 文档切分 → 向量化 → Milvus入库 → 知识图谱构建
```

### 查询流程

```
用户提问 → Query改写 → 向量检索 → Rerank重排 → 知识图谱查询 → RRF融合 → 生成回答
```

## 🔒 安全提醒

- **不要提交 `.env` 文件到 Git**（已配置在 `.gitignore` 中）
- 定期轮换 API Key 和密码
- 生产环境请修改默认密码

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请通过 Issue 反馈。

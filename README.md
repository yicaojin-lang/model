# 多大模型本地盲测与对比评估工具

完全本地化的多模型推理对比平台。所有测试数据、输入 Prompt 和模型输出均只存储在本机 SQLite 数据库中，**绝不向任何外部网络发送数据**。

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端框架 | FastAPI + asyncio |
| 数据库 | SQLite（aiosqlite 异步驱动） |
| ORM | SQLAlchemy 2.0（异步模式） |
| 大模型网关 | 本地 Ollama（`http://127.0.0.1:11434`） |
| HTTP 客户端 | httpx（异步） |
| 前端 UI | Jinja2 Templates + TailwindCSS CDN + Alpine.js CDN |

## 项目结构

```
modelTool/
├── app/
│   ├── main.py               # FastAPI 入口，lifespan 钩子
│   ├── config.py             # 全局配置（Settings dataclass）
│   ├── database.py           # SQLAlchemy 异步引擎 & 会话工厂
│   ├── models/
│   │   └── orm.py            # SQLAlchemy ORM 模型
│   ├── schemas/
│   │   └── api.py            # Pydantic v2 请求/响应 schema
│   ├── services/
│   │   ├── ollama.py         # Ollama HTTP 客户端（keep_alive=0）
│   │   └── runner.py         # 后台评测任务执行器
│   ├── routers/
│   │   ├── api_benchmarks.py # REST API：测试集
│   │   ├── api_runs.py       # REST API：评测任务
│   │   ├── api_responses.py  # REST API：评分
│   │   ├── api_models.py     # REST API：Ollama 模型列表
│   │   └── web.py            # Web UI Jinja2 路由
│   └── templates/            # HTML 模板
│       ├── base.html
│       ├── index.html
│       ├── benchmarks/
│       └── runs/
└── data/                     # SQLite 数据库文件（自动创建）
```

## 快速启动

### 1. 前置条件

- Python 3.11+
- [Ollama](https://ollama.com) 已安装并在本机运行

```bash
ollama serve          # 确保 Ollama 正在运行
ollama pull llama3    # 拉取至少一个模型
```

### 2. 安装依赖

```bash
cd modelTool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 启动服务

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Web UI**：http://127.0.0.1:8000
- **REST API 文档**：http://127.0.0.1:8000/docs

## 核心功能

### 测试集（Benchmark）

- 创建包含多个 Prompt 的测试集，可选填写参考答案
- 支持删除（级联删除所有评测数据）

### 评测任务（Evaluation Run）

- 从本地 Ollama 模型列表中勾选 N 个模型
- 任务在后台异步执行：**逐模型顺序推理**，每次推理结束立即释放显存（`keep_alive=0`），防止统一内存 OOM
- 实时进度轮询（前端每 2 秒查询一次）

### 盲测对比（Blind Compare）

- 模型名称默认匿名（显示为「模型 A」「模型 B」…）
- 点击星级（1–5）即时保存评分，无需二次确认
- 点击「揭示模型名称」显示真实模型 ID 及性能指标（延迟 / token 数 / 推理速度）
- 评分汇总面板实时显示各模型平均分排名

## REST API 速览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/models` | 列出本地 Ollama 模型 |
| GET/POST | `/api/benchmarks` | 列出 / 创建测试集 |
| GET/DELETE | `/api/benchmarks/{id}` | 获取 / 删除测试集 |
| GET/POST | `/api/runs` | 列出 / 创建并启动评测任务 |
| GET | `/api/runs/{id}` | 获取任务状态与进度 |
| GET | `/api/runs/{id}/responses` | 获取全部模型响应 |
| GET | `/api/runs/{id}/stats` | 获取评分统计 |
| DELETE | `/api/runs/{id}` | 删除任务 |
| POST | `/api/responses/{id}/score` | 提交 / 更新评分 |

## 硬件说明

本工具专为 **Mac Mini（统一内存架构）** 优化：

- `keep_alive=0`：每次推理完成后 Ollama 立即卸载模型，释放统一内存
- 逐模型串行推理：不同时加载两个大模型，杜绝 OOM
- 完全异步 I/O：等待推理期间不阻塞服务器处理其他请求

## 数据安全

- 数据库文件：`data/modelTool.db`（纯本地 SQLite）
- 所有网络请求仅访问 `127.0.0.1:11434`（本机 Ollama）
- 无任何遥测、无日志上报、无外部依赖项在运行时联网

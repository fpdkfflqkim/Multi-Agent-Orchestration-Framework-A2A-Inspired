# 🤖 Multi-Agent Orchestration Framework (A2A-inspired)

This project implements a multi-agent orchestration system where an LLM analyzes user requests, generates an execution DAG, and distributes tasks across specialized agents.

Each request is processed through an end-to-end pipeline:

1. Agent launch & health check
2. Agent discovery (Agent Cards)
3. DAG generation by LLM
4. Task execution:
   - parallel execution -> `asyncio.gather`
   - sequential execution -> dependency ordering
5. Answer generation
6. Evaluation scoring
7. Iterative refinement loop (if needed)

The system enables:
- Dynamic task routing
- Modular agent composition
- Iterative response optimization via evaluation feedback

## 🗺️ Framework Architecture
![alt text](<Multi-Agent Orchestration Framework (A2A-inspired).png>)
## ⚙️ Installation

### Prerequisites

| Tool | Purpose |
| --- | --- |
| `Python 3.12+` | Runtime environment |
| [`uv`](https://docs.astral.sh/uv/) | Dependency management |
| [`Ollama`](https://ollama.com) | Local LLM serving |
| [`Docker`](https://www.docker.com) | Required for Neo4j & Langfuse |

### Setup

**1. Install dependencies**
```bash
$ uv sync
```

**2. Docker Compose**

```bash
$ docker-compose up -d
```

- **Neo4j** — Graph database for GraphRAG retrieval (http://localhost:7474)
- **Langfuse** — LLM observability and tracing dashboard (http://localhost:3000)

**3. Pull Ollama Models**

Pull the LLM and embedding models used in this project.  
Model names must match those defined in `src/config.py`.

```bash
# LLM  (set MODEL_NAME in src/config.py)
$ ollama pull {MODEL_NAME}
 
# Embedding model  (set EMBEDDING_MODEL in src/config.py)
$ ollama pull {EMBEDDING_MODEL}
```

**4. Environment variables**
Create a `.env` file in the project root:

```bash
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_SECRET_KEY= # your_secret_key
LANGFUSE_PUBLIC_KEY= # your_public_key
LANGFUSE_HOST=http://localhost:3000

NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME= # your_username
NEO4J_PASSWORD= # your_password
```

**5. Index Documents (RAG Agent)**
Place source documents (`.pdf` or `.txt`) in `src/db/input_doc/`, then run:

```bash
python -m src.db.{graphdb|vectordb}.indexing
```
This extracts entity triples via LLM, stores them in Neo4j, and creates vector embeddings for hybrid retrieval.

## 🧰 Tech Stack

| Category | Stack |
| --- | --- |
| Language | `Python 3.12` |
| Dependency manager | `uv` |
| LLM | `Ollama` |
| Containerization | `Docker` |
| Agent Framework | `LangChain` |
| API Server | `FastAPI`, `Uvicorn` |
| Graph Database | `Neo4j` |
| MCP Framework | `FastMCP` |
| Observability | `Langfuse` |

## 🚀 Usage Guide

### Running the Framework

| Module | Description | Run Command |
|---|---|---|
| `/src/agent_mono.py` | Run a single agent with MCP tools attached | `$ python -m src.agent_mono`
| `/src/orchestrator.py` | Full multi-agent orchestration (DAG-based) | `$ python -m src.orchestrator`

Running `orchestrator.py` automatically launches all registered agent servers as subprocesses — no manual startup required.
 
To run agents individually:
 
```bash
$ python -m src.agents.<agent_name>
```

#### > Configuration(`src/config.py`)

| Parameter | Description |
|---|---|
| `MODEL_NAME` | LLM used across the orchestrator and all agents |
| `EMBEDDING_MODEL` | Embedding model used for vector operations |
| `TEMPERATURE` | Controls generation randomness |
| `TIMEOUT` | Health check and request timeout per agent (seconds) |
| `EVAL_MAX_RETRY` | Maximum number of Answer → Eval refinement iterations |
| `MAX_ROUTING_RETRY` | Maximum retries for DAG plan parsing on LLM output failure |
| `AGENTS_INFO` | Registry of all agents — port, script path, and MCP server path |

---

## 🧩 Agent Overview
Agents are divided into two categories: **domain agents**, which handle specific task types and are selected dynamically by the orchestrator, and **pipeline agents**, which run in a fixed order after all domain results are collected.

### Domain Agents
 
Stateless, domain-specific execution units that perform deterministic tasks such as computation and data retrieval. Each agent exposes its capabilities as MCP tools and operates within a clearly bounded domain.
 
| Agent | Port | Description |
|---|---|---|
| `math_agent` | `8001` | Mathematical computation — local functions exposed as MCP tools |
| `weather_agent` | `8002` | Weather data retrieval — local functions exposed as MCP tools |
| `finance_agent` | `8003` | Financial market data — external API (Yahoo Finance) wrapped as MCP tools |
| `graphrag_agent` | `8006` | Hybrid knowledge retrieval — Graph DB (entity relationships) + Vector DB (semantic similarity) |
 
### Pipeline Agents
 
Responsible for aggregating domain results, generating a final response, and evaluating its quality. These agents carry no domain-specific tools — they operate purely on text.
 
| Agent | Port | Description |
|---|---|---|
| `answer_agent` | `8004` | Synthesizes all domain agent outputs into a coherent final response |
| `eval_agent` | `8005` | Scores the generated answer and returns structured feedback (score / pass / feedback).<br>Triggers re-generation if the quality threshold is not met |

## 🔁 Orchestration Flow

The orchestrator processes each request through four stages:
 
**1. Agent Launch & Health Check**  
All agents listed in `AGENTS_INFO` are spawned as subprocesses. The orchestrator polls each agent's `GET /.well-known/agent.json` endpoint until it responds with HTTP 200, up to the configured `TIMEOUT`.
 
**2. Agent Discovery**  
The orchestrator fetches each agent's Agent Card, which contains its name, description, and skill definitions. This information is injected into the routing prompt at runtime.
 
**3. DAG Planning**  
The LLM receives the collected agent descriptions and the user's request, then returns a structured JSON execution plan. Tasks with no inter-dependencies are grouped into the same step and executed concurrently via `asyncio.gather()`, where each task execution issues a `POST /tasks/send` request. Tasks that depend on prior results are placed in subsequent steps and receive the accumulated context as input.
 
**4. Answer & Eval Loop**  
Once all domain agents have completed, their outputs are passed to the Answer Agent, which synthesizes a final response. The Eval Agent then scores the response and returns structured feedback. If the response does not meet the quality threshold, the feedback is passed back to the Answer Agent for refinement. This loop continues until the answer passes evaluation or `EVAL_MAX_RETRY` is exhausted — in which case the last generated answer is returned.

## 📁 Project Structure

```bash
Project Root
├── src/
│   ├── agents/
│   │   ├── answer_agent.py
│   │   ├── eval_agent.py
│   │   ├── finance_agent.py
│   │   ├── graphrag_agent.py
│   │   ├── math_agent.py
│   │   └── weather_agent.py
│   ├── db/                         # Database clients and retrieval logic
│   ├── mcp_servers/
│   │   ├── graphrag_server.py      # MCP server exposing Graph DB
│   │   ├── math_server.py          # MCP server exposing math functions
│   │   └── weather_server.py       # MCP server exposing weather functions
│   ├── orchestrator.py
│   ├── agent_mono.py
│   └── config.py
├── .env
├── docker-compose.yml
├── neo4j_auth.txt  
├── pyproject.tomls
└── uv.lock
```
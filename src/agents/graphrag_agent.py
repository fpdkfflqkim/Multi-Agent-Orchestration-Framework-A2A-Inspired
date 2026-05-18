"""
graphrag Agent - A2A Server (port 8006)
graphrag MCP 서버에 연결된 LangChain 에이전트를 A2A 프로토콜로 노출

Google A2A spec:
  GET  /.well-known/agent.json  → Agent Card 반환
  POST /tasks/send              → 동기 태스크 처리

"""
# python -m src.agents.weather_agent

import os
from pathlib import Path
import uuid
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import CONFIG as cfg

from dotenv import load_dotenv
load_dotenv()
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

# ── A2A Agent Card ──────────────────────────────────────────────────────────
AGENT_CARD = {
    "name": "GraphRAG Agent",
    "description": "그래프DB와 벡터DB를 활용해 질문에 답변하는 에이전트입니다.",
    "url": f"http://localhost:{cfg.AGENTS_INFO['graphrag']['port']}",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },
    "skills": [
        {
            "id": "graphrag_query",
            "name": "GraphRAG 질의응답",
            "description": "그래프/벡터 검색 결과를 바탕으로 질문에 답변합니다.",
            "examples": ["00에 대해 알려줘", "A와 B의 관계는?"],
        }
    ],
}

# ── FastAPI 앱 ───────────────────────────────────────────────────────────────
agent_executor: AgentExecutor | None = None


def build_agent(tools: list) -> AgentExecutor:
    llm = ChatOllama(model=cfg.MODEL_NAME, temperature=cfg.TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant with access to a knowledge graph and vector database. "
            "Always use the provided tools to retrieve relevant information before answering. "
            "Your answer MUST be based solely on the retrieved data. "
            "If the retrieved data does not contain relevant information, respond with '해당 정보를 찾을 수 없습니다.' "
            "Do NOT use your own knowledge or make assumptions beyond what is retrieved. "
            "Always respond in Korean.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

PROJECT_ROOT = str(Path(__file__).resolve().parents[2]) 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    print("[GraphRAG Agent] MCP 서버 연결 중...")
    client = MultiServerMCPClient({
        "graphrag": {
            "command": "python",
            "args": [cfg.AGENTS_INFO["graphrag"]["mcp_server"]],
            "transport": "stdio",
            "env": {
                **os.environ,
                "PYTHONPATH": PROJECT_ROOT,
        }
        }
    })
    tools = await client.get_tools()
    agent_executor = build_agent(tools)
    print(f"[GraphRAG Agent] 준비 완료 — port {cfg.AGENTS_INFO['graphrag']['port']}")
    yield

    langfuse_handler._langfuse_client.flush()
    print("[GraphRAG Agent] 종료 중...")


app = FastAPI(title="GraphRAG A2A Agent", lifespan=lifespan)


# ── A2A 엔드포인트 ────────────────────────────────────────────────────────────

@app.get("/.well-known/agent.json")
async def get_agent_card():
    return JSONResponse(content=AGENT_CARD)


@app.post("/tasks/send")
async def tasks_send(body: dict[str, Any]):
    task_id = body.get("id", str(uuid.uuid4()))
    parts = body.get("message", {}).get("parts", [])
    user_text = " ".join(p.get("text", "") for p in parts if "text" in p)

    if not user_text:
        return _error_response(task_id, "메시지 텍스트가 없습니다.")

    try:
        result = await agent_executor.ainvoke(
            {"input": user_text},
            config={
                "callbacks": [langfuse_handler],
                "run_name": "GraphRAG Agent",
            }
        )
        answer = result.get("output", "")
        return _success_response(task_id, answer)
    except Exception as e:
        return _error_response(task_id, str(e))


# ── 응답 헬퍼 ────────────────────────────────────────────────────────────────

def _success_response(task_id: str, text: str) -> dict:
    return {
        "id": task_id,
        "status": {"state": "completed"},
        "artifacts": [{"parts": [{"type": "text", "text": text}], "index": 0}],
    }


def _error_response(task_id: str, error: str) -> dict:
    return {
        "id": task_id,
        "status": {
            "state": "failed",
            "message": {"role": "agent", "parts": [{"text": error}]},
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=cfg.AGENTS_INFO["graphrag"]["port"])

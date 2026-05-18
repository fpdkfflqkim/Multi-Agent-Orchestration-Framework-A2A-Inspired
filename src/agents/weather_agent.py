"""
Weather Agent - A2A Server (port 8002)
Weather MCP 서버에 연결된 LangChain 에이전트를 A2A 프로토콜로 노출

Google A2A spec:
  GET  /.well-known/agent.json  → Agent Card 반환
  POST /tasks/send              → 동기 태스크 처리
#    POST /tasks/sendSubscribe
"""
# python -m src.agents.weather_agent

import asyncio
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
    "name": "Weather Agent",
    "description": "특정 도시의 현재 날씨를 조회하는 에이전트입니다.",
    "url": f"http://localhost:{cfg.AGENTS_INFO['weather']['port']}",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },
    "skills": [
        {
            "id": "current_weather",
            "name": "현재 날씨 조회",
            "description": "도시 이름을 받아 현재 날씨(기온)를 반환합니다.",
            "examples": ["서울 날씨 알려줘", "도쿄 지금 몇 도야?"],
        }
    ],
}

# ── FastAPI 앱 ───────────────────────────────────────────────────────────────
agent_executor: AgentExecutor | None = None


def build_agent(tools: list) -> AgentExecutor:
    """tools를 받아 LangChain 에이전트를 생성합니다."""
    llm = ChatOllama(model=cfg.MODEL_NAME,
                     temperature=cfg.TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a weather assistant. Use the provided tools to get weather information. "
                "Always respond in Korean.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    print("[Weather Agent] MCP 서버 연결 중...")
    client = MultiServerMCPClient(
        {
            "weather": {
                "command": "python",
                "args": [cfg.AGENTS_INFO["weather"]["mcp_server"]],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    agent_executor = build_agent(tools)
    print("[Weather Agent] 준비 완료 — port 8002")
    yield
    
    langfuse_handler._langfuse_client.flush()
    print("[Weather Agent] MCP 서버 종료 중...")


app = FastAPI(title="Weather A2A Agent", lifespan=lifespan)


# ── A2A 엔드포인트 ────────────────────────────────────────────────────────────

@app.get("/.well-known/agent.json")
async def get_agent_card():
    """A2A Agent Discovery."""
    return JSONResponse(content=AGENT_CARD)


@app.post("/tasks/send")
async def tasks_send(body: dict[str, Any]):
    """A2A tasks/send 엔드포인트 (동기)."""
    task_id = body.get("id", str(uuid.uuid4()))
    message = body.get("message", {})
    parts = message.get("parts", [])
    user_text = " ".join(p.get("text", "") for p in parts if "text" in p)

    if not user_text:
        return _error_response(task_id, "메시지 텍스트가 없습니다.")

    try:
        result = await agent_executor.ainvoke({"input": user_text},
                                              config={
                                                  "callbacks": [langfuse_handler],
                                                  "run_name": f"Weather Agent",
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
        "artifacts": [
            {"parts": [{"type": "text", "text": text}], "index": 0}
        ],
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
    uvicorn.run(app, host="0.0.0.0", port=cfg.AGENTS_INFO["weather"]["port"])

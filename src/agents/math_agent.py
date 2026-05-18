"""
Math Agent - A2A Server (port 8001)
Math MCP 서버에 연결된 LangChain 에이전트를 A2A 프로토콜로 노출

Google A2A spec:
  GET  /.well-known/agent.json  → Agent Card 반환
  POST /tasks/send              → 동기 태스크 처리
"""
# python -m src.agent.math_agent

import asyncio
import uuid
from datetime import datetime
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
    "name": "Math Agent",
    "description": "섭씨/화씨 변환 등 수학 계산을 수행하는 Agent",
    "url": f"http://localhost:{cfg.AGENTS_INFO['math']['port']}",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },
    "skills": [
        {
            "id": "temperature_conversion",
            "name": "온도 변환",
            "description": "섭씨(Celsius)를 화씨(Fahrenheit)로 변환",
            "examples": ["30도를 화씨로 변환해줘", "100°C는 몇 °F야?"],
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
                "You are a math assistant. Use the provided tools for calculations. "
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
    print("[Math Agent] MCP 서버 연결 중...")
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [cfg.AGENTS_INFO["math"]["mcp_server"]],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    agent_executor = build_agent(tools)
    print("[Math Agent] 준비 완료 — port 8001")
    yield
    
    langfuse_handler._langfuse_client.flush()
    print("[Math Agent] MCP 서버 종료 중...")


app = FastAPI(title="Math A2A Agent", lifespan=lifespan)


# ── A2A 엔드포인트 ────────────────────────────────────────────────────────────

@app.get("/.well-known/agent.json")
async def get_agent_card():
    """A2A Agent Discovery: Agent Card를 반환합니다."""
    return JSONResponse(content=AGENT_CARD)


@app.post("/tasks/send")
async def tasks_send(body: dict[str, Any]):
    """
    A2A tasks/send 엔드포인트 (동기).

    요청 형식 (A2A spec):
    {
        "id": "<task_id>",
        "message": {
            "role": "user",
            "parts": [{"text": "사용자 메시지"}]
        }
    }
    """
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
                                                  "run_name": f"Math Agent",
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
            {
                "parts": [{"type": "text", "text": text}],
                "index": 0,
            }
        ],
    }


def _error_response(task_id: str, error: str) -> dict:
    return {
        "id": task_id,
        "status": {"state": "failed", "message": {"role": "agent", "parts": [{"text": error}]}},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=cfg.AGENTS_INFO["math"]["port"])

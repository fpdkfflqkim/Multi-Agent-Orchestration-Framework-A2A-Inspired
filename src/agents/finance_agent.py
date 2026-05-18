"""
Finance Agent - A2A Server (port 8003)
Finance MCP 서버에 연결된 LangChain 에이전트를 A2A 프로토콜로 노출

Google A2A spec:
  GET  /.well-known/agent.json  → Agent Card 반환
  POST /tasks/send              → 동기 태스크 처리
"""
# python -m src.agents.finance_agent

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
    "name": "Finance Agent",
    "description": "Yahoo Finance 데이터를 조회하는 에이전트입니다. 주가, 재무제표, 배당, 기업 관련 뉴스 등을 제공합니다.",
    "url": f"http://localhost:{cfg.AGENTS_INFO['finance']['port']}",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },
    "skills": [
        {
            "id": "stock_price",
            "name": "주가 조회",
            "description": "특정 종목의 현재 주가 및 과거 주가 데이터를 조회합니다.",
            "examples": ["애플 주가 알려줘", "TSLA 현재 주가", "삼성전자 최근 1달 주가 변동"],
        },
        {
            "id": "financials",
            "name": "재무 정보 조회",
            "description": "손익계산서, 현금흐름표, 배당 정보, 실적 발표일을 조회합니다.",
            "examples": ["AAPL 손익계산서", "MSFT 배당 정보", "테슬라 실적 발표일"],
        },
        {
            "id": "news_and_recommendations",
            "name": "뉴스 및 추천",
            "description": "종목 관련 최신 뉴스와 애널리스트 추천 정보를 조회합니다.",
            "examples": ["NVDA 최신 뉴스", "구글 애널리스트 추천"],
        },
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
                "You are a financial data assistant. "
                "Use the provided Yahoo Finance tools to retrieve accurate financial information. "
                "When asked about stock prices, always use the ticker symbol (e.g. AAPL, TSLA, MSFT). "
                "Always respond in Korean, but keep ticker symbols in English.",
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
    print("[Finance Agent] MCP 서버 연결 중...")
    client = MultiServerMCPClient(
        {
            "finance": {
                "command": "uvx",
                "args": ["mcp-yahoo-finance"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    agent_executor = build_agent(tools)
    print(f"[Finance Agent] 준비 완료 — port {cfg.AGENTS_INFO["finance"]["port"]} (툴 {len(tools)}개 로드됨)")
    yield
    
    langfuse_handler._langfuse_client.flush()
    print("[Finance Agent] MCP 서버 종료 중...")
 
 
app = FastAPI(title="Finance A2A Agent", lifespan=lifespan)
 
 
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
                                                  "run_name": f"Finance Agent",
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
    uvicorn.run(app, host="0.0.0.0", port=cfg.AGENTS_INFO["finance"]["port"])
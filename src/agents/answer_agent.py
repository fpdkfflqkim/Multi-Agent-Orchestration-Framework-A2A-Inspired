"""
Answer Agent - A2A Server (port 8004)
Answer MCP 서버에 연결된 LangChain 에이전트를 A2A 프로토콜로 노출

Google A2A spec:
  GET  /.well-known/agent.json  → Agent Card 반환
  POST /tasks/send              → 동기 태스크 처리
  
입력 (JSON 문자열로 전달):
  {
    "user_input":    "원본 사용자 질문",
    "agent_results": "[math]\n결과...\n\n[weather]\n결과...",
    "feedback":      "(선택) 평가 에이전트의 피드백 — 재생성 시 전달"
  }
"""
# python -m src.agents.answer_agent

import json
import uuid
from typing import Any
from contextlib import asynccontextmanager
 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
 
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
 
from src.config import CONFIG as cfg

from dotenv import load_dotenv
load_dotenv()
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

# ── A2A Agent Card ──────────────────────────────────────────────────────────
AGENT_CARD = {
    "name": "Answer Agent",
    "description": "여러 에이전트의 결과를 통합해 자연스러운 한국어 답변을 생성합니다.",
    "url": f"http://localhost:{cfg.AGENTS_INFO['answer']['port']}",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },
    "skills": [
        {
            "id": "synthesize",
            "name": "결과 통합 답변 생성",
            "description": "다수 에이전트 결과와 사용자 질문을 받아 하나의 자연스러운 답변으로 통합합니다.",
            "examples": [
                "날씨와 수학 계산 결과를 합쳐서 답변해줘",
                "여러 에이전트 결과를 자연스럽게 정리해줘",
            ],
        }
    ],
}

# ── LLM ────────────────────────────────────────────────────────────────────────
 
llm: ChatOllama | None = None
 
SYSTEM_PROMPT = """당신은 여러 AI 에이전트의 결과를 사용자 친화적으로 통합하는 어시스턴트입니다.
 
규칙:
- 사용자의 질문에 직접 답하는 형태로 작성합니다.
- 에이전트 이름([math], [weather] 등) 같은 메타 정보는 노출하지 않습니다.
- 여러 결과가 있으면 자연스럽게 하나의 흐름으로 연결합니다.
- 불필요한 반복이나 군더더기 없이 간결하게 작성합니다.
- 평가 피드백이 있으면 해당 부분을 반드시 개선합니다.
- 항상 한국어로 답변합니다."""

# ── FastAPI 앱 ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    print("[Answer Agent] LLM 초기화 중...")
    llm = ChatOllama(model=cfg.MODEL_NAME, temperature=cfg.TEMPERATURE)
    print(f"[Answer Agent] 준비 완료 — port {cfg.AGENTS_INFO['answer']['port']}")
    yield
    
    langfuse_handler._langfuse_client.flush()
    print("[Answer Agent] 종료")
 
 
app = FastAPI(title="Answer A2A Agent", lifespan=lifespan)
 
 
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
    raw_text = " ".join(p.get("text", "") for p in parts if "text" in p)
 
    if not raw_text:
        return _error_response(task_id, "메시지 텍스트가 없습니다.")
 
    # JSON 페이로드 파싱
    try:
        payload = json.loads(raw_text)
        user_input = payload.get("user_input", "")
        agent_results = payload.get("agent_results", "")
        feedback = payload.get("feedback", "")
    except (json.JSONDecodeError, TypeError):
        # 단순 텍스트 폴백
        user_input = raw_text
        agent_results = raw_text
        feedback = ""
 
    if not user_input:
        return _error_response(task_id, "user_input이 없습니다.")
 
    try:
        answer = await _synthesize(user_input, agent_results, feedback)
        return _success_response(task_id, answer)
    except Exception as e:
        return _error_response(task_id, str(e))
 
# ── 핵심 로직 ──────────────────────────────────────────────────────────────────
 
async def _synthesize(user_input: str, agent_results: str, feedback: str = "") -> str:
    feedback_section = (
        f"\n\n[이전 답변에 대한 평가 피드백 — 반드시 반영하세요]\n{feedback}"
        if feedback else ""
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"[사용자 질문]\n{user_input}\n\n"
            f"[에이전트 결과들]\n{agent_results}"
            f"{feedback_section}"
        )),
    ]
    resp = await llm.ainvoke(messages,
                             config={
                                 "callbacks": [langfuse_handler],
                                 "run_name": f"Answer Agent",
                                 }
                             )
    return resp.content.strip()
 
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
    uvicorn.run(app, host="0.0.0.0", port=cfg.AGENTS_INFO["answer"]["port"])
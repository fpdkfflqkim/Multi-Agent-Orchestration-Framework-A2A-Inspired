"""
eval Agent - A2A Server (port 8005)
eval MCP 서버에 연결된 LangChain 에이전트를 A2A 프로토콜로 노출

Google A2A spec:
  GET  /.well-known/agent.json  → Agent Card 반환
  POST /tasks/send              → 동기 태스크 처리
  
입력 (JSON 문자열로 전달):
  {
    "user_input":    "원본 사용자 질문",
    "agent_results": "도메인 에이전트 원본 결과 (사실 검증용)",
    "answer":        "Answer Agent가 생성한 최종 답변"
  }
 
응답 artifacts (JSON 문자열):
  {
    "passed":   true | false,
    "score":    0~10,
    "feedback": "개선이 필요한 내용 (passed=true면 빈 문자열)"
  }
"""
# python -m src.agents.eval_agent

import json
import uuid
from typing import Any
from contextlib import asynccontextmanager
 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
 
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
 
from src.config import CONFIG as cfg

from dotenv import load_dotenv
load_dotenv()
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

# ── A2A Agent Card ──────────────────────────────────────────────────────────
AGENT_CARD = {
    "name": "Eval Agent",
    "description": "답변의 정확성·완전성·자연스러움을 평가하고 통과 여부와 피드백을 반환합니다.",
    "url": f"http://localhost:{cfg.AGENTS_INFO['eval']['port']}",
    "version": "1.0.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },
    "skills": [
        {
            "id": "evaluate",
            "name": "답변 품질 평가",
            "description": "사용자 질문, 원본 에이전트 결과, 최종 답변을 비교해 품질을 0~10점으로 평가합니다.",
            "examples": [
                "이 답변이 질문에 제대로 답하고 있는지 평가해줘",
                "답변의 정확성과 완전성을 확인해줘",
            ],
        }
    ],
}

# ── LLM ────────────────────────────────────────────────────────────────────────
PASS_THRESHOLD = cfg.EVAL_PASS_THRESHOLD
llm: ChatOllama | None = None
 
SYSTEM_PROMPT = f"""당신은 AI 답변 품질 평가자입니다.
사용자 질문, 에이전트 원본 결과, 최종 답변을 분석해 아래 기준으로 평가하세요.
 
평가 기준:
1. 정확성     — 에이전트 결과의 핵심 수치·사실이 답변에 올바르게 반영되었는가?
2. 완전성     — 사용자 질문의 모든 부분에 답하고 있는가?
3. 자연스러움 — 한국어 표현이 자연스럽고 군더더기 없이 간결한가?
4. 메타 정보  — 에이전트 이름([math], [weather] 등) 같은 내부 정보가 노출되지 않았는가?
 
반드시 아래 JSON 형식만 반환 (다른 텍스트 금지):
{{
  "passed": true 또는 false,
  "score": 0~10 사이 정수,
  "feedback": "개선이 필요한 구체적인 내용 (passed=true면 빈 문자열)"
}}
 
passed 기준: score >= {PASS_THRESHOLD}"""

# ── FastAPI 앱 ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    print("[Eval Agent] LLM 초기화 중...")
    llm = ChatOllama(model=cfg.MODEL_NAME, temperature=0)   # 평가는 temperature=0 고정
    print(f"[Eval Agent] 준비 완료 — port {cfg.AGENTS_INFO['eval']['port']}")
    yield
    
    langfuse_handler._langfuse_client.flush()
    print("[Eval Agent] 종료")
 
 
app = FastAPI(title="Eval A2A Agent", lifespan=lifespan)
 
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
 
    try:
        payload = json.loads(raw_text)
        user_input = payload.get("user_input", "")
        agent_results = payload.get("agent_results", "")
        answer = payload.get("answer", "")
    except (json.JSONDecodeError, TypeError):
        return _error_response(task_id, "페이로드 JSON 파싱 실패")
 
    if not answer:
        return _error_response(task_id, "평가할 answer가 없습니다.")
 
    try:
        eval_result = await _evaluate(user_input, agent_results, answer)
        return _success_response(task_id, json.dumps(eval_result, ensure_ascii=False))
    except Exception as e:
        return _error_response(task_id, str(e))
    
# ── 핵심 로직 ──────────────────────────────────────────────────────────────────
 
async def _evaluate(user_input: str, agent_results: str, answer: str) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"[사용자 질문]\n{user_input}\n\n"
            f"[에이전트 원본 결과]\n{agent_results}\n\n"
            f"[최종 답변]\n{answer}"
        )),
    ]
    resp = await llm.ainvoke(messages,
                            config={
                                 "callbacks": [langfuse_handler],
                                 "run_name": f"Eval Agent",
                                 }
                            )
    raw = resp.content.strip()
    print(f"[DDDDDDDEBUG] last_trace_id: {langfuse_handler.last_trace_id}")
 
    # JSON 펜스 제거
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
 
    try:
        result = json.loads(raw.strip())
        result.setdefault("passed", False)
        result.setdefault("score", 0)
        result.setdefault("feedback", "")
        return result
    except (json.JSONDecodeError, TypeError):
        return {
            "passed": False,
            "score": 0,
            "feedback": f"평가 파싱 실패 (원본: {raw[:200]}). 답변을 다시 생성해 주세요.",
        }
 
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
    uvicorn.run(app, host="0.0.0.0", port=cfg.AGENTS_INFO["eval"]["port"])
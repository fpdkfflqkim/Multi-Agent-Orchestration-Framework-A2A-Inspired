"""
오케스트레이터 (A2A Client)
사용자 요청을 분석해 Math Agent 또는 Weather Agent로 라우팅합니다.

흐름:
  1. AGENT_URLS에 등록된 에이전트 서버를 subprocess로 자동 실행
  2. 각 에이전트가 준비될 때까지 헬스체크 (최대 30초)
  3. LLM으로 어느 에이전트가 요청을 처리할지 결정
  4. 해당 에이전트의 /tasks/send 에 A2A 태스크 전송
  5. 오케스트레이터 종료 시 모든 자식 프로세스 자동 종료
"""
# python -m src.orchestrator

import asyncio
import json
import signal
import subprocess
import sys
import uuid
from urllib.parse import urlparse
from dataclasses import dataclass, field

import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
load_dotenv()

from src.config import CONFIG as cfg
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent 


# ── 에이전트 엔드포인트 + 실행 스크립트 설정 ─────────────────────────────────
AGENT_URLS = {
    "math":    {"url": f"http://localhost:{cfg.AGENTS_INFO['math']['port']}",    "script": cfg.AGENTS_INFO["math"]["agent"]},
    "weather": {"url": f"http://localhost:{cfg.AGENTS_INFO['weather']['port']}", "script": cfg.AGENTS_INFO["weather"]["agent"]},
    "finance": {"url": f"http://localhost:{cfg.AGENTS_INFO['finance']['port']}", "script": cfg.AGENTS_INFO["finance"]["agent"]},
    "graphrag": {"url": f"http://localhost:{cfg.AGENTS_INFO['graphrag']['port']}", "script": cfg.AGENTS_INFO["graphrag"]["agent"]},

    "answer":  {"url": f"http://localhost:{cfg.AGENTS_INFO['answer']['port']}",  "script": cfg.AGENTS_INFO["answer"]["agent"]},
    "eval":    {"url": f"http://localhost:{cfg.AGENTS_INFO['eval']['port']}",    "script": cfg.AGENTS_INFO["eval"]["agent"]},
}

# ── 실행 계획 데이터 구조 ─────────────────────────────────────────────────────



NOT_DOMAIN_AGENTS = {"answer", "eval"} # 라우터 할당하지 않는 에이전트
DOMAIN_AGENTS = AGENT_URLS.keys() - NOT_DOMAIN_AGENTS


@dataclass
class AgentTask:
    """단일 에이전트 실행 단위"""
    agent: str 
    query: str
    result: str = "" 


@dataclass
class ExecutionStep:
    tasks: list[AgentTask] = field(default_factory=list)


# ── AgentLauncher ─────────────────────────────────────────────────────────────

class AgentLauncher:
    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}

    def launch_all(self):
        for name, agent_cfg in AGENT_URLS.items():
            script = agent_cfg.get("script")
            if not script:
                continue
            module = script.replace("/", ".").removesuffix(".py")
            print(f"    ✓ [{name}] Agent Launch: {module}")
        
            proc = subprocess.Popen(
                [sys.executable, "-m", module],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            
            self._procs[name] = proc


    async def wait_until_ready(self, timeout: float = cfg.TIMEOUT):
        async with httpx.AsyncClient(timeout=timeout) as http:
            for name, agent_cfg in AGENT_URLS.items():
                url = agent_cfg["url"]
                health_url = f"{url}/.well-known/agent.json"
                deadline = asyncio.get_event_loop().time() + timeout
                # print(f"   [{name}] ({url})...", end="", flush=True)
                while True:
                    try:
                        r = await http.get(health_url)
                        if r.status_code == 200:
                            print(f"    ✓ [{name}] ({url}) is ready.")
                            break
                    except Exception:
                        pass
                    if asyncio.get_event_loop().time() > deadline:
                        print(f"    ✗ [{name}] ({url}) is timeout ")
                        raise RuntimeError(f"    ✗ [{name}] did not respond within {timeout} seconds.")
                    await asyncio.sleep(0.5)

    def terminate_all(self):
        for name, proc in self._procs.items():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                # print(f"  ✓ [{name}] is terminated")
        self._procs.clear()


# ── A2AClient ─────────────────────────────────────────────────────────────────

class A2AClient:
    def __init__(self, timeout: float = cfg.TIMEOUT):
        self.http = httpx.AsyncClient(timeout=timeout)

    async def discover(self, base_url: str) -> dict:
        resp = await self.http.get(f"{base_url}/.well-known/agent.json")
        resp.raise_for_status()
        return resp.json()

    async def send_task(self, base_url: str, user_text: str) -> str:
        payload = {
            "id": str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": user_text}],
            },
        }
        resp = await self.http.post(f"{base_url}/tasks/send", json=payload)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status", {}).get("state") == "completed":
            artifacts = data.get("artifacts", [])
            if artifacts:
                parts = artifacts[0].get("parts", [])
                return " ".join(p.get("text", "") for p in parts if "text" in p)
        elif data.get("status", {}).get("state") == "failed":
            error_parts = data.get("status", {}).get("message", {}).get("parts", [])
            error_text = " ".join(p.get("text", "") for p in error_parts)
            raise RuntimeError(f"에이전트 오류: {error_text}")

        return str(data)

    async def close(self):
        await self.http.aclose()


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    DAG 기반 A2A 오케스트레이터 (v3).

    도메인 에이전트 DAG 실행 후 Answer → Eval 루프로 품질 보장.
    """

    def __init__(self):
        self.launcher = AgentLauncher()
        self.client = A2AClient()
        self.llm = ChatOllama(model=cfg.MODEL_NAME, temperature=cfg.TEMPERATURE)
        self.agent_cards: dict[str, dict] = {}
        self.langfuse_handler = CallbackHandler()
        
    async def startup(self):
        langfuse = Langfuse()
        print("Langfuse 연결 확인:", langfuse.auth_check())
        # self.langfuse_handler.langfuse = langfuse

        print("🔧 Agent Environment Setup ───────────────────────────────────────────────\n")
        print("  🚀 Agent launch ===") # 에이전트 서버 실행
        self.launcher.launch_all()

        print("\n  ⏳ Agent Ready Check ===") # 헬스 확인
        await self.launcher.wait_until_ready(timeout=cfg.TIMEOUT)

        print("\n  🔍 Agent Discovery ===") # 에이전트 카드 조회
        for name, agent_cfg in AGENT_URLS.items():
            url = agent_cfg["url"]
            try:
                card = await self.client.discover(url)
                self.agent_cards[name] = card
                print(f"    ✓ [{name}] : {card['description']}") # print(f"  ✓ [{name}] {card['name']} : {card['description']}")
            except Exception as e:
                print(f"    ✗ [{name}] Failed to discover card: {type(e).__name__}: {e}")
        print("\n✅ Setup Complete ───────────────────────────────────────────────\n")

    # ── 라우팅 프롬프트 ───────────────────────────────────────────────────────

    def _build_routing_prompt(self) -> str:
        agent_descriptions = []
        for key, card in self.agent_cards.items():
            if key not in DOMAIN_AGENTS: # == if key in NOT_DOMAIN_AGENTS:
                continue
            skills = card.get("skills", [])
            skill_desc = ", ".join(s["description"] for s in skills)
            agent_descriptions.append(
                f'- "{key}": {card["name"]} — {card["description"]} (skiils: {skill_desc})'
            )
        agents_text = "\n".join(agent_descriptions)

        return f"""당신은 DAG 실행 플래너입니다. 사용자 요청을 분석해 에이전트 실행 계획을 JSON으로 반환하세요.

사용 가능한 에이전트:
{agents_text}

규칙:
- 서로 독립적인 작업은 같은 step의 agents 배열에 묶어 병렬 실행합니다.
- 이전 step의 결과가 필요한 작업은 다음 step에 배치합니다.
- 각 agent의 query는 해당 에이전트가 처리할 구체적인 질의문을 작성합니다.
- 에이전트가 다른(이전 단계의) 에이전트 결과를 필요로 하는 경우, 해당 결과를 명시적으로 참조할 수 있도록 쿼리에 "이전 단계의 [결과값]을 [작업]" 형태의 지시를 포함한다.

반드시 아래 JSON 형식만 반환 (다른 텍스트 금지):
{{
  "steps": [
    {{
      "agents": [
        {{"agent": "에이전트명", "query": "이 에이전트에 전달할 질의"}}
      ]
    }}
  ],
  "reason": "실행 계획 선택 이유"
}}"""

    # ── 실행 계획 파싱 ────────────────────────────────────────────────────────

    def _parse_plan(self, raw: str, user_input: str) -> list[ExecutionStep]:
        """LLM 응답에서 ExecutionStep 리스트를 파싱."""
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw.strip())
            steps = []
            for step_data in data.get("steps", []):
                tasks = []
                for a in step_data.get("agents", []):
                    agent_name = a.get("agent", "").lower()
                    if agent_name in DOMAIN_AGENTS:
                        tasks.append(AgentTask(agent=agent_name, query=a.get("query", user_input)))
                if tasks:
                    steps.append(ExecutionStep(tasks=tasks))
            if steps:
                return steps
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        return None

    # ── 단일 AgentTask 실행 ───────────────────────────────────────────────────

    async def _run_task(self, task: AgentTask, context: str = "") -> AgentTask:

        query = task.query
        if context:
            query = f"[이전 단계 결과]\n{context}\n\n[현재 질의]\n{task.query}"

        url = AGENT_URLS[task.agent]["url"]
        try:
            task.result = await self.client.send_task(url, query)
        except Exception as e:
            task.result = f"[오류] {type(e).__name__}: {e}"
        return task

    # ── DAG 실행 ─────────────────────────────────────────────────────────────

    async def _execute_dag(self, steps: list[ExecutionStep]) -> str:
        context_parts: list[str] = []

        for step_idx, step in enumerate(steps):
            agent_names = [t.agent for t in step.tasks]
            parallel_str = "병렬" if len(step.tasks) > 1 else "단독"
            print(f"    Step {step_idx+1} {parallel_str} 실행: {agent_names}")

            context = "\n".join(context_parts)

            completed: list[AgentTask] = await asyncio.gather(
                *[self._run_task(task, context) for task in step.tasks]
            )

            for task in completed:
                print(f"      ✓ [{task.agent}] response : {task.result[:30]}{'...' if len(task.result) > 30 else ''}")
                # print(task.result)
                context_parts.append(f"[{task.agent}]\n{task.result}")

        return "\n\n".join(context_parts)

    # ── Answer → Eval 루프 ────────────────────────────────────────────────────

    async def _answer_eval_loop(self,
                                user_input: str,
                                agent_results: str) -> str:

        answer_url = AGENT_URLS["answer"]["url"]
        eval_url = AGENT_URLS["eval"]["url"]
        feedback = ""
        answer = "죄송합니다. 답변 생성에 실패했습니다."
        
        MAX_RETRY = cfg.EVAL_MAX_RETRY

        for attempt in range(1, MAX_RETRY + 1):
            print(f"    [Attempt {attempt}/{MAX_RETRY}]")
            # ── 1. 답변 생성 ──
            print(f"      Answer 생성")
            answer_payload = json.dumps({
                "user_input": user_input,
                "agent_results": agent_results,
                "feedback": feedback,
            }, ensure_ascii=False)

            try:
                answer = await self.client.send_task(answer_url, answer_payload)
                # answer = "가나다라마바사아자차카타파하" # 수정 확인
            except Exception as e:
                print(f"      ✗ Answer Agent 오류: {type(e).__name__}: {e}")
                continue  

            # print(f"    답변 미리보기: {answer[:100]}{'...' if len(answer) > 100 else ''}")

            # ── 2. 품질 평가 ──
            print(f"      Answer 평가")
            eval_payload = json.dumps({
                "user_input": user_input,
                "agent_results": agent_results,
                "answer": answer,
            }, ensure_ascii=False)

            try:
                eval_raw = await self.client.send_task(eval_url, eval_payload)
                eval_result = json.loads(eval_raw)
            except Exception as e:
                print(f"      ✗ Eval Agent 오류: {type(e).__name__}: {e} — 현재 답변을 반환합니다.")
                return answer

            score = eval_result.get("score", 0)
            passed = eval_result.get("passed", False)
            feedback = eval_result.get("feedback", "")

            if passed:
                print(f"        ✓ pass") # print(f"      ✓ pass (점수: {score}/10)")
                # print(f"      Evaluation passed ({attempt} attempts)")
                return answer
            else:
                print("        ✗ fail")
                print(f"        feedback: {feedback[:120]}{'...' if len(feedback) > 120 else ''}")


        # MAX_RETRY 소진 → 마지막 답변 반환
        print(f"    ⚠️  최대 재시도({MAX_RETRY}회) 소진. 마지막 답변을 반환합니다.")
        return answer
    
    
    # ── 메인 라우팅 ──────────────────────────────────────────────────────────

    async def route(self, user_input: str) -> str:
        if not self.agent_cards:
            return "에이전트가 연결되지 않았습니다."

        # Step 1: LLM으로 DAG 실행 계획 수립
        messages = [
            SystemMessage(content=self._build_routing_prompt()),
            HumanMessage(content=user_input),
        ]
        print("  📋 PLAN")
        steps = None
        MAX_ROUTING_RETRY = cfg.MAX_ROUTING_RETRY
        for attempt in range(1, MAX_ROUTING_RETRY + 1):
            routing_resp = await self.llm.ainvoke(messages,
                                                  config={"callbacks": [self.langfuse_handler],
                                                          "run_name": f"Client Agent"
                                                          }
                                                  )
            raw = routing_resp.content.strip()
            steps = self._parse_plan(raw, user_input)

            if steps:
                break
             
            print(f"    ⚠️  라우팅 실패 ({attempt}/{MAX_ROUTING_RETRY})")
            print(f"      {raw}")
            # print(f"    ⚠️  라우팅 실패 ({attempt}/{MAX_ROUTING_RETRY}): {raw[:80]}...")
            messages.append(AIMessage(content=raw))
            messages.append(HumanMessage(content=(
                "응답이 올바른 JSON 형식이 아닙니다. "
                "반드시 지정된 JSON 형식만 반환해주세요. 다른 텍스트는 포함하지 마세요."
            )))

        if not steps:
            print(f"    ✗ {MAX_ROUTING_RETRY}회 재시도 후에도 실행 계획 파싱 실패")
            failure_msg = "죄송합니다. 요청을 처리할 적절한 에이전트를 찾지 못했습니다. 질문을 다르게 표현해 주시겠어요?"
            return failure_msg

        # 계획 출력
        for i, step in enumerate(steps):
            names = [t.agent for t in step.tasks]
            print(f"    Step {i+1}: {names} {'(병렬)' if len(names) > 1 else ''}")

        print("\n  ⚙️  EXECUTION")
        agent_results = await self._execute_dag(steps)

        print("\n  🔁 Answer & Eval")
        return await self._answer_eval_loop(user_input, agent_results)

    async def shutdown(self):
        await self.client.close()
        self.langfuse_handler._langfuse_client.flush()
        print("\n🏁 COMPLETE ───────────────────────────────────────────────")
        self.launcher.terminate_all()


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

async def main():
    orchestrator = Orchestrator()
    await orchestrator.startup()
    
    print("\n🚀 REQUEST START ───────────────────────────────────────────────\n")
    queries = [
        # "가나다라마바사아자차카타파하", # 라우팅 확인용
        "업계에서 부정적인 평가를 받고 있는 회사와 그 이유는?",
        # "30도를 화씨로 변환해줘",
        # "테슬라 최신 뉴스 알려줘",
        # "도쿄는 지금 몇 도야? 그리고 화씨로도 알려줘",
        # "도쿄의 기온을 화씨로 알려주고, 애플 관련 최신 뉴스도 같이 알려줘",
    ]
    
    try:
        for query in queries:
            print(f"  👤 USER INPUT : {query}\n")
            try:
                answer = await orchestrator.route(query)
                print(f"\n  🤖 FINAL ANSWER:\n{answer}")
                print()
            except Exception as e:
                print(f"❌ 오류: {e}")
    except KeyboardInterrupt:
        print("\n\n중단 요청 수신...")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import CONFIG as cfg

async def main():
    # 1. MCP 클라이언트 설정 (기존과 동일)
    PROJECT_ROOT = str(Path(__file__).resolve().parents[1]) 
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [cfg.AGENTS_INFO["math"]["mcp_server"]],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": [cfg.AGENTS_INFO["weather"]["mcp_server"]],
                "transport": "stdio",
            },
            "finance": {
                "command": "uvx",
                "args": ["mcp-yahoo-finance"],
                "transport": "stdio",
            },
            "graphrag": {
                "command": "python",
                "args": [cfg.AGENTS_INFO["graphrag"]["mcp_server"]],
                "transport": "stdio",
                "env": {
                    **os.environ,
                    "PYTHONPATH": PROJECT_ROOT,
                }
            },
        }
    )

    # 2. 서버로부터 도구 목록 가져오기
    tools = await client.get_tools()

    # 3. 모델 설정 (툴 호출 기능을 지원하는 모델이어야 함)
    llm = ChatOllama(model=cfg.MODEL_NAME,
                     temperature=cfg.TEMPERATURE)

    # 4. 표준 Tool Calling 프롬프트
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Always use the provided tools to answer questions accurately."),
        ("human", "{input}"),
        # 툴 호출 기록과 결과가 들어가는 중요한 플레이스홀더
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 5. 표준 툴 호출 에이전트 생성
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    # 6. 실행
    print("에이전트 실행")
    query = "업계에서 부정적인 평가를 받고 있는 회사와 그 이유는?"
    
    result = await agent_executor.ainvoke({"input": query})

    print("\n✨ 최종 답변:")
    print(result["output"])

if __name__ == "__main__":
    asyncio.run(main())
class CONFIG:
    MODEL_NAME = "gemma4:31b-cloud"
    EMBEDDING_MODEL = "embeddinggemma"
    TEMPERATURE = 0
    TIMEOUT = 30.0
    EVAL_PASS_THRESHOLD = 7
    MAX_ROUTING_RETRY = 3
    EVAL_MAX_RETRY = 3 
    
    AGENTS_INFO = {
        "math": {
            "port": 8001,
            "agent": "src/agents/math_agent.py",
            "mcp_server": "src/mcp_servers/math_server.py"
        },
        "weather": {
            "port": 8002,
            "agent": "src/agents/weather_agent.py",
            "mcp_server": "src/mcp_servers/weather_server.py"
        },
        "finance": {
            "port": 8003,
            "agent": "src/agents/finance_agent.py",
            "mcp_server": "mcp-yahoo-finance"
        },
        "answer": {
            "port": 8004,                          # 기존 포트와 겹치지 않게 설정
            "agent": "src/agents/answer_agent.py",
            },
        "eval": {
            "port": 8005,
            "agent": "src/agents/eval_agent.py",
            },
        "graphrag": {
            "port": 8006,
            "agent": "src/agents/graphrag_agent.py",
            "mcp_server": "src/mcp_servers/graphrag_server.py"
            },
        }

from fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_current_weather(city: str) -> str:
    """특정 도시의 현재 기온을 가져옵니다. (현재는 30도 고정)"""
    return f"{city}의 현재 기온은 섭씨 30도입니다."

if __name__ == "__main__":
    mcp.run(transport="stdio")
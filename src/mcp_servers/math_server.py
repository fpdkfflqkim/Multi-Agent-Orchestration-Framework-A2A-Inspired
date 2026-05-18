from fastmcp import FastMCP

mcp = FastMCP("math")

@mcp.tool()
def celsius_to_fahrenheit(celsius: float) -> str:
    """섭씨 온도를 화씨 온도로 변환합니다."""
    fahrenheit = (celsius * 9/5) + 32
    return f"{celsius}°C는 화씨로 {fahrenheit}°F입니다."

if __name__ == "__main__":
    mcp.run(transport="stdio")
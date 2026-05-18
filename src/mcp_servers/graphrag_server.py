from fastmcp import FastMCP
from src.db.graphdb.retrieve import graph_retriever, vector_retriever

mcp = FastMCP("graphrag")

@mcp.tool()
def graph_retrieve(query: str) -> str:
    """그래프 DB에서 엔티티 관계를 검색합니다."""
    graph_data, entities = graph_retriever(query)
    return graph_data

@mcp.tool()
def vector_retrieve(query: str, k: int = 5) -> str:
    """벡터 DB에서 유사 문서를 검색합니다."""
    vector_data = vector_retriever(query, k=k)
    return "\n-----\n".join(vector_data)

if __name__ == "__main__":
    mcp.run(transport="stdio")
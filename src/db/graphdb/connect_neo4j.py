# python -m src.db.graphdb.connect_neo4j
import os
from dotenv import load_dotenv

from src.db.graphdb.config import MyConfig as mcfg

from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_ollama import OllamaEmbeddings
load_dotenv()

def connect_n4j_graph():
    try:
        graph = Neo4jGraph(
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD")
        )
        print("Connected NEO4J Graph")
        return graph
    except Exception as e:
        print("Not Connected NEO4J Graph")
        print(e)

def connect_n4j_vector():
    embed = OllamaEmbeddings(model=mcfg.embd_model)
    try:
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embed,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="vector",
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD")
            )
        print("Connected NEO4J Vector")
        return vector_index
    except Exception as e:
        print("Not Connected NEO4J Vector")
        print(e)

if __name__ == "__main__":
    print("* testing connect_neo4j")
    connect_n4j_graph()
    connect_n4j_vector()
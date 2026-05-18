# python -m src.db.graphdb.retrieve
import os
import re
import json

from src.db.graphdb.config import MyConfig as mcfg
from src.db.graphdb.connect_neo4j import connect_n4j_graph, connect_n4j_vector
from src.db.graphdb.prompts import GraphRAG_Prompt, make_prompt
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama



class Entities(BaseModel):
        entities: list[str] = Field(description="질문에서 언급되는 모든 개체")

def graph_retriever(query,
                    llm_model=mcfg.ollama_model):
    graph = connect_n4j_graph()
    llm = ChatOllama(model=llm_model)
    prompts=GraphRAG_Prompt()
    
    # indexing
    q_extraction_prompt = make_prompt(sys_prompt=prompts.retrieve_sys_prompt,
                                      human_prompt=prompts.retrieve_human_prompt,
                                      format_instructions=prompts.retrieve_output_format)
    
    entity_chain = q_extraction_prompt | llm
    response = entity_chain.invoke({"input": query})
    cleaned = re.sub(r"```json|```", "", response.content).strip()
    entities = Entities(**json.loads(cleaned))
    # print(entities)
    graph_results = []
    
    for entity in entities.entities:
        query_response = graph.query(
            """
            MATCH (p)-[r*1..3]-(e)
            WHERE p.id CONTAINS $entity
            AND ALL(rel IN r WHERE type(rel) <> 'MENTIONS')

            WITH 
            p, e,
            [rel IN r | { 
                type: type(rel), 
                start: startNode(rel).id, 
                end: endNode(rel).id 
            }] AS rel_list

            WITH 
            p.id AS start_node,
            e.id AS end_node,
            apoc.coll.toSet(apoc.coll.flatten(collect(rel_list))) AS rels

            RETURN 
            [start_node, end_node] AS node_ids,
            rels
            LIMIT 50""",
            {"entity": entity}
        )
        
    
        # graph_data += "\n".join([f"{el['source_id']} - {el['relationship']} -> {el['target_id']}" for el in query_response])
        graph_results.extend(query_response)
        
    graph_data = "\n".join(str(item) for item in graph_results)
    return graph_data , entities

def vector_retriever(query, k=5):
    vector_index = connect_n4j_vector()
    sim_docs = vector_index.similarity_search(
        query=query,
        k=k,
        # filter={"dept": "HR"}
        )
    vector_data = [f"Document{idx}{el.page_content}" for idx,el in enumerate(sim_docs, start=1)]
    
    return vector_data
    
        
if __name__ == "__main__":
    print("* testing graphrag.retrieve")
    
    query = "박정호에 대해 알려줘"
    graph_data, entities = graph_retriever(query)
    vector_data = vector_retriever(query, k=3)
    print(graph_data)
    print(vector_data)

    # full_data = f"[Graph data]\n{graph_data}\n\n[Related Documents]\n{"\n-----\n".join(vector_data)}"
    # print(full_data)
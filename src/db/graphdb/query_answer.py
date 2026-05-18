# python -m src.db.graphdb.query_answer
import os
import re
import json

from src.db.graphdb.config import MyConfig as mcfg
from src.db.graphdb.prompts import GraphRAG_Prompt, make_prompt
from src.db.graphdb.retrieve import graph_retriever, vector_retriever
from langchain_ollama import ChatOllama


def query_answering(query,
                    llm_model=mcfg.ollama_model):
    llm = ChatOllama(model=llm_model)
    prompts=GraphRAG_Prompt()
    
    graph_data, entities = graph_retriever(query)
    all_data = f"[Graph data]\n{graph_data}"
    
    query_answering_prompt = make_prompt(sys_prompt=prompts.qna_sys_prompt,
                                  human_prompt=prompts.qna_human_prompt,
                                  context=all_data)
    
    qna_chain = query_answering_prompt | llm
    response = qna_chain.invoke({"input": query})

    return response

def query_answering_hybrid(query,
                         llm_model=mcfg.ollama_model,
                         k=5):
    llm = ChatOllama(model=llm_model)
    prompts=GraphRAG_Prompt()
    
    graph_data, entities = graph_retriever(query)
    vector_data = vector_retriever(query, k=k)
    all_data = f"[Graph data]\n{graph_data}\n\n[Related Documents]\n{"\n-----\n".join(vector_data)}"
    
    query_answering_prompt = make_prompt(sys_prompt=prompts.qna_sys_prompt_hybrid,
                                  human_prompt=prompts.qna_human_prompt,
                                  context=all_data)
    
    qna_chain = query_answering_prompt | llm
    response = qna_chain.invoke({"input": query})

    return response
    
        
if __name__ == "__main__":
    print("* testing graph query & answer")
    
    query = "박정호에 대해 알려줘"

    response = query_answering(query)
    response_hybrid = query_answering(query, k=3)
    print(response)
    print(response_hybrid)
# python -m src.db.graphdb.indexing
import os

from src.db.graphdb.config import MyConfig as mcfg
from src.db.graphdb.prompts import GraphRAG_Prompt, make_prompt

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer

def indexing(graph,
             documents,
             llm_model=mcfg.ollama_model,
             embed_model=mcfg.embd_model):
    llm = ChatOllama(model=llm_model)
    embed = OllamaEmbeddings(model=embed_model)
    prompts=GraphRAG_Prompt()
    
    # indexing
    indexing_prompt = make_prompt(prompts.index_sys_prompt,
                                  prompts.index_human_prompt,
                                  prompts.index_example,
                                  prompts.index_output_format)
    
    llm_transformer_filtered = LLMGraphTransformer(llm=llm,
                                                   prompt=indexing_prompt,
                                                   ignore_tool_usage=True
                                                   )
    
    # text = documents[0].page_content
    # raw_schema = llm_transformer_filtered.chain.invoke({"input": text})
    # print(raw_schema)
    
    graph_documents = llm_transformer_filtered.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents,
                              baseEntityLabel=True,
                              include_source=True)
    
    for gdoc in graph_documents:
        doc_id = gdoc.source.metadata["id"]
        vector = embed.embed_documents(gdoc.source.page_content)[0] # []
        
        graph.query("""
            MATCH (d:Document {id: $doc_id})
            SET d.vector = $vector
        """, {"doc_id": doc_id, "vector": vector})
    
        
if __name__ == "__main__":
    print("* testing graph indexing")
    
    from src.db.graphdb.connect_neo4j import connect_n4j_graph
    from src.db.graphdb.data_processing import data_processing, documents_dict
    from src.db.graphdb.config import MyConfig as mcfg
    
    graph = connect_n4j_graph()
    origin_docs = documents_dict()
    fin_docs = data_processing(origin_docs, 300, 50)
    
    indexing(graph, fin_docs)
    
    # graph.query("MATCH (n) DETACH DELETE n")
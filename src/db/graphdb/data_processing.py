# python -m src.db.graphdb.data_processing

import os

import pymupdf4llm
from src.db.graphdb.config import MyConfig as mcfg
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def documents_dict(path=mcfg.data_path):
    folder_path = path
    files = os.listdir(folder_path)
    file_vars = {}

    for file_name in files:
        file_ex = os.path.splitext(file_name)[-1]
        file_path = os.path.join(folder_path, file_name)
        if file_ex == ".pdf":
            initial_content = pymupdf4llm.to_markdown(file_path)
            content = "".join(initial_content)
        if file_ex == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        key_name = os.path.splitext(file_name)[0]
        file_vars[key_name] = content
        
    return file_vars

def data_processing(doc_dict,
                    chunk_size=500,
                    chunk_overlap=50):
    
    all_documents = []
    
    # chunking
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                 chunk_overlap=chunk_overlap)
    for doc_title, doc_content in doc_dict.items():
        chunks = text_splitter.split_text(doc_content)
        
        for idx, chunk in enumerate(chunks, start=1):
            doc = Document(
                page_content=chunk,
                metadata={
                    "id": f"{doc_title}-{idx}",
                    "source": doc_title
                    }
                )

            all_documents.append(doc)
    return all_documents
    


if __name__ == "__main__":
    print("* testing graph data_processing")
    a = documents_dict()
    print(a)
    b = data_processing(a, 300, 50)
    print(b)


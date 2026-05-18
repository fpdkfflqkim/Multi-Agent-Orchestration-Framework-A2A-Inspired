# python -m src.db.graphdb.config
from src.config import CONFIG as cfg

class MyConfig:
    data_path = "./src/db/input_doc"
    prompt_path = "./src/db/graphdb/prompt"
    ollama_model = cfg.MODEL_NAME
    embd_model = cfg.EMBEDDING_MODEL
    
    
if __name__ == "__main__":
    print("* testing graphrag.config")
    print(MyConfig.prompt_path)
    print(MyConfig.ollama_model)
    print(MyConfig.embd_model)
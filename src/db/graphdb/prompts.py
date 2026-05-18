# python -m src.db.graphdb.prompts

import os
from src.db.graphdb.config import MyConfig as mcfg

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

class GraphRAG_Prompt:
    def __init__(self, folder_path=mcfg.prompt_path):
        self.folder_path = folder_path
        
        self.index_example = self.read_prompt(self.folder_path, "index_example.txt")
        self.index_human_prompt = self.read_prompt(self.folder_path, "index_human_prompt.txt")
        self.index_output_format = self.read_prompt(self.folder_path, "index_output_format.txt")
        self.index_sys_prompt = self.read_prompt(self.folder_path, "index_sys_prompt.txt")
        
        self.retrieve_human_prompt = self.read_prompt(self.folder_path, "retrieve_human_prompt.txt")
        self.retrieve_output_format = self.read_prompt(self.folder_path, "retrieve_output_format.txt")
        self.retrieve_sys_prompt = self.read_prompt(self.folder_path, "retrieve_sys_prompt.txt")
        
        self.qna_sys_prompt = self.read_prompt(self.folder_path, "qna_sys_prompt.txt")
        self.qna_sys_prompt_hybrid = self.read_prompt(self.folder_path, "qna_sys_prompt_hybrid.txt")
        self.qna_human_prompt = self.read_prompt(self.folder_path, "qna_human_prompt.txt")
        
        
        
        
    def read_prompt(self,
                    folder_path,
                    prompt_name):
        
        prompt_path = os.path.join(folder_path, prompt_name)
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return content
    
def make_prompt(sys_prompt,
                human_prompt,
                examples=None,
                format_instructions=None,
                context=None):
    # sys_prompt
    SYS_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)
    
    # human_prompt
    
    partial_variables = {}
    if examples:
        partial_variables["examples"] = examples
    if format_instructions:
        partial_variables["format_instructions"] = format_instructions
    if context:
        partial_variables["context"] = context
        
    HUMAN_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["input"],
            partial_variables=partial_variables,
            template=human_prompt
            )
        )
    fin_prompt = ChatPromptTemplate.from_messages([SYS_prompt, HUMAN_prompt])
    
    return fin_prompt
        
if __name__ == "__main__":
    print("* testing graph prompts")
    prompts=GraphRAG_Prompt()
    print(prompts.index_human_prompt)
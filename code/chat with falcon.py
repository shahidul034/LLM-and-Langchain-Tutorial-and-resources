import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from typing import Set
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from typing import Any, List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import textwrap
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
import re
import warnings
from typing import List
 
import torch
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFacePipeline
from langchain.schema import BaseOutputParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
 
warnings.filterwarnings("ignore", category=UserWarning)

def llm_return():
    path=r'/home/drmohammad/data/llm/falcon-7b-instruct'
    tokenizer = AutoTokenizer.from_pretrained(path,
                                            use_auth_token=True,)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, load_in_8bit=True, device_map="auto"
    )
    generation_config = model.generation_config
    generation_config.temperature = 0
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 256
    generation_config.use_cache = False
    generation_config.repetition_penalty = 1.7
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    
    class StopGenerationCriteria(StoppingCriteria):
        def __init__(
            self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
        ):
            stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
            self.stop_token_ids = [
                torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
            ]
    
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            for stop_ids in self.stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                    return True
            return False
    stop_tokens = [["Human", ":"], ["AI", ":"]]
    stopping_criteria = StoppingCriteriaList([StopGenerationCriteria(stop_tokens, tokenizer, model.device)])
    generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    stopping_criteria=stopping_criteria,
    generation_config=generation_config,
    )
 
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
        
    return llm

llm=llm_return()
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)
while True:
    inp=str(input("Human: "))
    if inp == "1":
        break
    res=llm_chain.predict(human_input=inp)
    print("*"*20,res)
    



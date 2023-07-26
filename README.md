
# LLM training and Langchain tutorial

A brief description of what this project does and who it's for LLM training ,testing and integrate with Langchain.


## Installation

```
conda create -n lang python=3.8
conda activate lang
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
python -m pip install jupyter
pip install langchain
pip install streamlit
pip install streamlit-chat
```
## Load any model locally
```
path=r'/home/drmohammad/Documents/LLM/Llamav2hf/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(path,
                                          use_auth_token=True,)

model = AutoModelForCausalLM.from_pretrained(path,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                            #  load_in_8bit=True,
                                            #  load_in_4bit=True
                                             )


pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
    
llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
```
## Create any custom prompt

```
instruction = "context:\n\n{context} \n\nQuestion: {question} \n\n Answer:"
system_prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."
template = get_prompt(instruction, system_prompt)
prompt = PromptTemplate(
input_variables=["context", "question"], template=template)
```
## Run simple LLM model with prompt
```
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

question = "What is the capital of England?"

print(llm_chain.run(question))
```
## Run the LLM with HuggingFacePipeline/

```
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
print(local_llm('What is the capital of France? '))
```
## Load and use custom embedding model 
```
from langchain.embeddings import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)
vectorstore=FAISS.from_documents(docs,embeddings)
vectorstore.save_local('faiss_index_react')
new_vectorstore=FAISS.load_local("langchain_pyloader/vectorize",hf)
```
## Build langchain without memory
```
qa=RetrievalQA.from_chain_type(llm=local_llm,chain_type='stuff',retriever=new_vectorstore.as_retriever())
res=qa.run('what is malaria?')
```

## Build Conversation langchain with memory
```
qa = ConversationalRetrievalChain.from_llm(
       llm=llm, retriever=new_vectorstore.as_retriever()
    )
res=qa({"question": query, "chat_history":chat_history})
```
## Read the pdf and divide into chunk

```
pdf_path=r"/home/drmohammad/Documents/LLM/Malaria.pdf"
loader=PyPDFLoader(pdf_path)
documents=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=30,separator='\n')
docs=text_splitter.split_documents(documents=documents)
```
## Train any LLM model using autotrain
https://github.com/huggingface/autotrain-advanced
### Install autotrain 
```
pip install autotrain-advanced
```
### Run this command
```
!autotrain llm --train --project_name my-llm --model meta-llama/Llama-2-7b-hf --data_path "/home/drmohammad/Documents/LLM/llm testing/experiment" --train_split "train" --valid_split "test" --text_column "text" --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 10 --num_train_epochs 3 --trainer sft
```
### parameter description
https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/cli/run_llm.py

## How to load and chat with Llama2 7B 
https://colab.research.google.com/drive/1Ssg-fffeJ0LG0m3DoTofeLPvOUQyG1h3?usp=sharing#scrollTo=H0shki19igLy

## Simple chatbot using Llama2 7B 
```
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"

template = get_prompt(instruction, system_prompt)
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)
llm_chain.predict(user_input="Can you tell me about yourself.")
```









    

import sys
import os
sys.path.append(os.path.abspath('/home/mshahidul/'))
from gpu_selection import _gpu_selection_
import argparse
parser = argparse.ArgumentParser(description="Training...")
parser.add_argument("--finetune", type=str,default="False" ,help="finetune or inference")
parser.add_argument("--ques", type=str,default="The cat is on the roof." ,help="question to answer")
args = parser.parse_args()


from unsloth import FastLanguageModel
if args.finetune=="True":
    model_name = "/home/mshahidul/webiner/models/Qwen2.5-0.5B-Instruct_alpaca"
else:
    model_name = "unsloth/Qwen2.5-0.5B-Instruct"

max_seq_length,dtype,load_in_4bit = 4092,None ,True
model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction:
                    {}

                    ### Input:
                    {}

                    ### Response:
                    {}"""
prompt='Translate the following English text into Spanish naturally and accurately:'
# ques = "The cat is on the roof."
inputs = tokenizer(
    [
        alpaca_prompt.format(
            f"{prompt}", # instruction
            args.ques, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 256, temperature = 0.3, top_p = 0.9, top_k = 32,do_sample=True)
ans=tokenizer.batch_decode(outputs)
start_marker = '### Response:\n'
end_marker = '<|im_end|'
start_index = ans[0].find(start_marker) + len(start_marker)
end_index = ans[0].find(end_marker)
response = ans[0][start_index:end_index].strip()
print("*" * 30)
print(f"Model: {model_name}")
print(f"Input: {args.ques}")
print(f"Output: {response}")
print("*" * 30)
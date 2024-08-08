import transformers
import torch
import json
from tqdm import tqdm

model_id = "models/Meta-Llama-3.1-8B-Instruct"
turn_num=3
input_file="SC_prp/data/tmp_output.jsonl"
output_file="SC_prp/data/PRP_SC_output"

def tp_prompt(message):
    prompt=f'''
    
    Here are two examples:
    ---
    <Input>: The final answer is 5.
    <Output>: ##answer: 5
    <Input>: ##answer: $93,000
    <Output>: ##answer: 93000
    ---
    You need to transform the final answer into the form of '##answer: X'. You should not output anything else.
    <Input> {message}
    <Output>:
    '''

    return prompt

# 加载模型
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    #device=1,
    device_map="auto",
    max_length=512,
    
)
pipeline.tokenizer.pad_token_id=128001
pipeline.tokenizer.padding_side='left',

file1=[]


with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        file1.append(json.loads(line.strip()))


# 生成输出

questions=[]
problems=[]
answers=[]
# print(data)
for idx in range(len(file1)):
    q1=file1[idx]["question"]
    a1=file1[idx]["answer"]
    questions.append([{"role":"system","content": "After generating the answer, you should add '##answer:' and the final numerical answer at the end (digit only)"},{"role": "user", "content":tp_prompt(a1)}])
    problems.append(q1)
    answers.append(a1)
#print(questions)
outputs=pipeline(questions,batch_size=256)


with open(output_file+".jsonl", 'w', encoding='utf-8') as file:
    pass

with open(output_file+".jsonl", 'a', encoding='utf-8') as file:
    for idx in range(len(outputs)):
        chat=outputs[idx]
        new_data = {"question":problems[idx], "answer": chat[0]["generated_text"][-1]["content"]}
        file.write(json.dumps(new_data) + '\n')

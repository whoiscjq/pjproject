import transformers
import torch
import json
from tqdm import tqdm

model_id = "models/Meta-Llama-3.1-8B-Instruct"
turn_num=3
input_file1="SC_prp/data/tmp_first_output.jsonl"
input_file2="SC_prp/data/tmp_veri_output.jsonl"
output_file="SC_prp/data/tmp_output"

def tp_prompt(q,a1,v1,a2,v2,a3,v3):
    prompt=f'''
    questions:{q}
    Here are several answers and verfication for this question. 
    1.answer: <{a1}>
    verification: <{v1}>
    2.answer: <{a2}>
    verification: <{v2}>
    3.answer: <{a3}>
    verification: <{v3}>
    You should check its verification and reasoning to decide whether it is reasonable. 
    You should only output add '##answer:' and the most reasonable final numerical answer of the question at the end (digit only). 
    Do not ouput anything else.
    '''
    return prompt

# def tp_prompt(q,a,b,c):
#     prompt=f'''
#     questions:{q}
#     Here are several answers for this answer, you should output the most rational one:
#     1.{a}
#     2.{b}
#     3.{c}
#     '''
#     return prompt

# 加载模型
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    #device=1,
    device_map="auto",
    max_length=5102,
    
)
pipeline.tokenizer.pad_token_id=128001
pipeline.tokenizer.padding_side='left',

file1=[]
file2=[]

with open(input_file1, 'r', encoding='utf-8') as file:
    for line in file:
        file1.append(json.loads(line.strip()))

with open(input_file2, 'r', encoding='utf-8') as file:
    for line in file:
        file2.append(json.loads(line.strip()))

# 生成输出

questions=[]
problems=[]
answers=[]
# print(data)
for idx in range(int(len(file2)/3)):
    q1=file1[idx*3]["question"]
    a1=file1[idx*3]["answer"]
    v1=file2[idx*3]["answer"]
    a2=file1[idx*3+1]["answer"]
    v2=file2[idx*3+1]["answer"]
    a3=file1[idx*3+2]["answer"]
    v3=file2[idx*3+2]["answer"]

    questions.append([{"role":"system","content": "After generating the answer, you should add '##answer:' and the final numerical answer at the end (digit only)"}, {"role": "user", "content":tp_prompt(q1,a1,v1,a2,v2,a3,v3)}])
    problems.append(q1)
    answers.append(a1)
#print(questions)
outputs=pipeline(questions,batch_size=10)


with open(output_file+".jsonl", 'w', encoding='utf-8') as file:
    pass

with open(output_file+".jsonl", 'a', encoding='utf-8') as file:
    for idx in range(len(outputs)):
        chat=outputs[idx]
        new_data = {"question":problems[idx], "answer": chat[0]["generated_text"][-1]["content"]}
        file.write(json.dumps(new_data) + '\n')

with open(output_file+"_baseline"+".jsonl", 'w', encoding='utf-8') as file:
    pass

with open(output_file+"_baseline"+".jsonl", 'a', encoding='utf-8') as file:
    for idx in range(len(problems)):
        new_data = {"question":problems[idx], "answer": answers[idx] }
        file.write(json.dumps(new_data) + '\n')

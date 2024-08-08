import transformers
import torch
import json
from tqdm import tqdm

model_id = "models/Meta-Llama-3.1-8B-Instruct"
turn_num=3
output_file="SC/SC_output2"#".jsonl"

def tp_prompt(q,a,b,c):
    prompt=f'''
    questions:{q}
    Here are several answers for this question. You should check its verification part to decide whether it is reasonable.
    1.{a}
    2.{b}
    3.{c}
    You should check its verification part to decide whether it is reasonable. After that, you should only output add '##answer:' and the most reasonable final numerical answer at the end (digit only), for example(##answer: 5). Do not ouput anything else.
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
    max_length=4096,
    
)
pipeline.tokenizer.pad_token_id=128001
pipeline.tokenizer.padding_side='left',

data=[]

with open("SC/tmp_output2.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# 生成输出

questions=[]
problems=[]
answers=[]
# print(data)
for idx in range(int(len(data)/3)):
    q1=data[idx*3]["question"]
    a1=data[idx*3]["answer"]
    a2=data[idx*3+1]["answer"]
    a3=data[idx*3+2]["answer"]

    questions.append([{"role":"system","content": "After generating the answer, you should add '##answer:' and the final numerical answer at the end (digit only)"},{"role": "user", "content":tp_prompt(q1,a1,a2,a3)}])
    problems.append(q1)
    answers.append(a1)
#print(questions)
outputs=pipeline(questions,batch_size=16)


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

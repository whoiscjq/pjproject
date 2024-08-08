import transformers
import torch
import json
from tqdm import tqdm

model_id = "models/Meta-Llama-3.1-8B-Instruct"
turn_num=3
output_file="SC/tmp_output2.jsonl"

# 加载模型

def gen_prompt(message):
    prompt=f'''
    ##questions:{message}
    Let's work this out it a step by step to be sure we have the right answer. Moreover，you should follow the following methods.

    '''
    return prompt


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    max_length=1024,
    
)
pipeline.tokenizer.pad_token_id=128001
pipeline.tokenizer.padding_side='left',

data=[]
count=0
with open("dataset/eval.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        #data=[json.loads(line.strip())]
        # count=count+1
        # if count>20: break
        data.append(json.loads(line.strip()))

# 生成输出

questions=[]
answers=[]
problems=[]
# print(data)
for message in data:
    for _ in range(turn_num):
        questions.append([{"role":"system","content": "After generating the answer, you should add '##answer:' and the final numerical answer at the end (digit only)"},{"role": "user", "content":gen_prompt(message["question"])+"Let's think step by step."}])
    #questions.append([{"role": "user", "content":message["question"]}])
        answers.append(message["answer"])
        problems.append(message["question"])

#print(questions)
outputs=pipeline(questions,batch_size=128)


with open(output_file, 'w', encoding='utf-8') as file:
    pass

with open(output_file, 'a', encoding='utf-8') as file:
    for idx in range(len(outputs)):
        chat=outputs[idx]
        new_data = {"question":problems[idx], "answer": chat[0]["generated_text"][-1]["content"]}
        file.write(json.dumps(new_data) + '\n')

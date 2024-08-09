import transformers
import torch
import json
from tqdm import tqdm

model_id = "models/Meta-Llama-3.1-8B-Instruct"
turn_num=3
output_file="SC_prp/data/tmp_veri_output.jsonl"
input_file="SC_prp/data/tmp_first_output.jsonl"
# 加载模型

def gen_prompt(question,message):
    prompt=f'''
    ##question:{question}
    ##reasoning path :{message}
    You need to do the following steps to verify the problem:
    - Identify one numerical value in the problem conditions and mask (remove) it.
    - Using the final result obtained in reasoning path, recalculate the masked numerical value.
    - If recalculation is not possible with the current masked value, choose a different numerical value to mask and attempt the recalculation again.
    - Compare the recalculated numerical value with the original value in the problem.
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
with open(input_file, 'r', encoding='utf-8') as file:
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
countt=1
for message in data:
        questions.append([{"role": "user", "content":gen_prompt(message["question"],message["answer"])}])
    #questions.append([{"role": "user", "content":message["question"]}])
        answers.append(message["answer"])
        problems.append(message["question"])
        count=count+1
        if count>300:
            break

#print(questions)
outputs=pipeline(questions,batch_size=64)


with open(output_file, 'w', encoding='utf-8') as file:
    pass

with open(output_file, 'a', encoding='utf-8') as file:
    for idx in range(len(outputs)):
        chat=outputs[idx]
        new_data = {"question":problems[idx], "answer": chat[0]["generated_text"][-1]["content"]}
        file.write(json.dumps(new_data) + '\n')

import transformers
import torch
import json
from tqdm import tqdm

model_id = "models/Meta-Llama-3.1-8B-Instruct"

# 加载模型
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
count=10
with open("dataset/eval.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        #data=[json.loads(line.strip())]
        if count>10: break
        data.append(json.loads(line.strip()))

# 生成输出

questions=[]
answers=[]
# print(data)
for message in data:
    questions.append([{"role":"system","content": "After generating the answer, you should add '####' and the final number answer at the end (digit only)"},{"role": "user", "content":message["question"]}])
    answers.append(message["answer"])

#print(questions)
outputs=pipeline(questions,batch_size=4)

output_file="dataset/baseline_output_test.jsonl"
with open(output_file, 'w', encoding='utf-8') as file:
    pass

with open(output_file, 'a', encoding='utf-8') as file:
    for chat in outputs:
        new_data = {"question":chat[0]["generated_text"][0]["content"], "answer": chat[0]["generated_text"][-1]["content"]}
        file.write(json.dumps(new_data) + '\n')

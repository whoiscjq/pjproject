from transformers import AutoTokenizer
import transformers
import torch
import json
from tqdm import tqdm

model_id = "models/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# 加载模型
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0,
    max_length=512,
)
data=[]
pipeline.tokenizer.pad_token_id =2
count=0
with open("dataset/eval.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        #data=[json.loads(line.strip())]
        count=count+1
        data.append(json.loads(line.strip()))
        if count>32:
            break

# 生成输出

questions=[]
answers=[]
# print(data)
for message in data:
    questions.append([{"role":"system","content": "After generating the answer, you should add '####' and the final number answer at the end (digit only)"},{"role": "user", "content":message["question"]}])
    answers.append(message["answer"])

#print(questions)
outputs=pipeline(questions,batch_size=16)
# print(outputs)

#print(outputs)
output_file="dataset/mini_baseline_itern.jsonl"
with open(output_file, 'w', encoding='utf-8') as file:
    pass

with open(output_file, 'a', encoding='utf-8') as file:
    for chat in outputs:
        new_data = {"question":chat[0]["generated_text"][0]["content"], "answer": chat[0]["generated_text"][-1]["content"]}
        file.write(json.dumps(new_data) + '\n')

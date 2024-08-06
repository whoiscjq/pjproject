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
    device_map="auto",
    max_length=1028,
)
data=[]
pipeline.tokenizer.pad_token_id =2
with open("dataset/eval.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        
        numbers = re.findall(r'####\s*(\d+)', text)
        data.append(json.loads(line.strip()))

# 生成输出

questions=[]
answers=[]
# print(data)
for message in data:
    questions.append([{"role":"system","content": "After generating the answer, you should add '####' and the final number answer at the end (digit only)"},{"role": "user", "content":message["question"]}])
    #questions.append([{"role": "user", "content":message["question"]}])
    answers.append(message["answer"])

#print(questions)
outputs=tqdm(pipeline(questions,batch_size=256))
# print(outputs)

#print(outputs)
output_file="dataset/mini_baseline_itern.jsonl"
with open(output_file, 'w', encoding='utf-8') as file:
    pass

with open(output_file, 'a', encoding='utf-8') as file:
    for chat in outputs:
        new_data = {"question":chat[0]["generated_text"][1]["content"], "answer": chat[0]["generated_text"][-1]["content"]}
        file.write(json.dumps(new_data) + '\n')

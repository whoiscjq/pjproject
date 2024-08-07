from transformers import AutoTokenizer
import transformers
import torch
import json
import lmdeploy

model_id = "models/internlm2-chat-1_8b"
#tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# 加载模型
pipeline = lmdeploy.pipeline(model_id)
data=[]
pipeline.tokenizer.pad_token_id =2
with open("dataset/eval.jsonl", 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# 生成输出

questions=[]
answers=[]
# print(data)
for message in data:
    #questions.append([{"role":"system","content": "After generating the answer, you should add '####' and the final number answer at the end (digit only)"},{"role": "user", "content":message["question"]+"\n Let's think step by step"}])
    questions.append([{"role": "user", "content":message["question"]}])
    answers.append(message["answer"])

#print(questions)
outputs=pipeline(questions,batch_size=256)
# print(outputs)

#print(outputs)
output_file="dataset/baselibe_itern_output.jsonl"
with open(output_file, 'w', encoding='utf-8') as file:
    pass

with open(output_file, 'a', encoding='utf-8') as file:
    for i in range(len(outputs)):
        new_data = {"question":data[i]["question"], "answer": outputs[i].text}
        file.write(json.dumps(new_data) + '\n')

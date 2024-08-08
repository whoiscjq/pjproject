import json

save_path="SC_prp/data/PRP_SC_output.jsonl"
input_path="dataset/eval.jsonl"

mydata=[]
with open(save_path, 'r', encoding='utf-8') as f:
    for line in f:
        mydata.append(json.loads(line.strip()))

origindata=[]
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        origindata.append(json.loads(line.strip()))

for i in range(len(mydata)):
    origindata[i]["model_response"]=mydata[i]["answer"]

with open(input_path, 'w', encoding='utf-8') as f:
    for data in origindata:
        f.write(json.dumps(data) + '\n')
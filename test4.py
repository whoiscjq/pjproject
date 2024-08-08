import json

evals=[]
with open("dataset/eval.jsonl", "r") as f:
    for line in f:
        evals.append(json.loads(line))

result=[]
with open("/mnt/workspace/llm/prp/result/prp_output.jsonl","r") as f:
    for line in f:
        data=json.loads(line)
        result.append(data)
process=[]

for evals_data in evals:
    flag=0
    for result_data in result:
        if evals_data["question"]==result_data["question"] and flag != 1 :
            process.append({"question":evals_data["question"],"answer":result_data["answer"]})
            flag=1
            continue
    if flag == 0 :
        process.append({"question":evals_data["question"],"result":"Not found"})

with open("dataset/process.jsonl","w") as f:
    for line in process:
        f.write(json.dumps(line)+"\n")
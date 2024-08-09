import json
import re

def distil(file_name):
    outputs=[]
    with open(file_name+'.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            text=json.loads(line.strip())
            #numbers=re.findall(r'\d+', text["answer"])
            numbers=re.findall(r'####\s*(\d+)', text["answer"].replace("##answer:","####"))
            try:
                number=numbers[-1].strip()
                outputs.append(number)
            except:
                outputs.append("")
    return outputs
        
gt_file="dataset/eval"
#test_file = "/mnt/workspace/llm/dataset/prp_llama_output"
#test_file="dataset/baseline_itern_output"
test_file="SC_prp/data/PRP_SC_output"
gt_numbers=distil(gt_file)
test_numbers=distil(test_file)

total=0
cor=0

idx_list=[]
for i in range(len(test_numbers)):
    total=total+1
    if gt_numbers[i] == test_numbers[i]:
        cor=cor+1
    else:
        idx_list.append(i)

with open(test_file+".txt","a") as f:
    f.write(f"{cor}/{total} \n")
print(cor/total)

questions=[]
answers=[]
with open(gt_file+".jsonl",'r') as f:
    for line in f:
        data=json.loads(line)
        questions.append(data["question"])

with open(test_file+".jsonl",'r') as f:
    for line in f:
        data=json.loads(line)
        answers.append(data["answer"])

with open (test_file+"_diff.txt","a") as f:
    for idx in idx_list:
        tmp={"question":questions[idx],"answer":answers[idx]}
        f.write(json.dumps(tmp)+"\n")


import json
import re

def distil(file_name):
    outputs=[]
    with open(file_name+'.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            text=json.loads(line.strip())
            numbers=re.findall(r'####\s*(\d+)', text["answer"])
            try:
                number=numbers[-1].strip()
                outputs.append(number)
            except:
                outputs.append("")
    return outputs
        
gt_file="dataset/eval"
test_file="dataset/baseline_llama_output"
gt_numbers=distil(gt_file)
test_numbers=distil(test_file)

total=0
cor=0

for i in range(len(test_numbers)):
    total=total+1
    if gt_numbers[i] == test_numbers[i]:
        cor=cor+1

with open(test_file+".txt","w") as f:
    f.write(f"{cor}/{total} \n")
print(cor/total)
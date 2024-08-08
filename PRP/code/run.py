# -*- coding: utf-8 -*-
# python run.py --data_index 0
import warnings
warnings.filterwarnings('ignore')
import json
import os
from pprint import pprint
import argparse
from tqdm import tqdm
import utils, prompt

def count_braces(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content.count('}')

proxies = None
prompt_strategy = 'PRP'
model_name = "gpt-3.5-turbo"
num_iteration = 5
max_length = 1024
samples=[]

with open(f'dataset/eval.jsonl', 'r') as f:
    for line in f:
        json_obj=json.loads(line)
        samples.append(json_obj)

save_path = os.path.join('/mnt/workspace/llm/prp/result', f'prp_output2.jsonl')


problems = [sample.get('question') for sample in samples]



if not os.path.exists(save_path):
    add_idx = 0
else:
    add_idx = count_braces(save_path)

idx=0
for problem in tqdm(problems):
    if idx < add_idx:
        idx += 1
        continue
    process_record = {}
    process_record['problem'] = problem


    final_answer,full_answer, process_record = prompt.pipline(
        process_record, 
        problem.replace('?', ' ?') ,
        num_iteration, 
        model_name, 
        max_length, 
        proxies
    )

    process_record['final_answer'] = final_answer
    with open(save_path, 'a', encoding='utf-8') as f:
        pass
    with open(save_path, 'a', encoding='utf-8') as f:
        #f.write(json.dumps(process_record, ensure_ascii=False) + '\n')
        data={"question":problem,"answer":full_answer}
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    

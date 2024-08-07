# -*- coding: utf-8 -*-
import time
import json
import re
from pprint import pprint
import transformers
import torch
import json
from tqdm import tqdm

#sleep_time = 20
SHOW = True

model_id = "models/Meta-Llama-3.1-8B-Instruct"

# 加载模型
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    max_length=512,
    
)
# def construct_verify_problem(problem):
#     prompt = f'''
#     Here are two examples:
#     --------
#     Input : "The serum will make the drinker grow an extra arm every three days and an extra leg every five days."
#     Output: three 

#     Input: "He creates a movie for $2000. Each DVD cost $6 to make.
#     Output: 2000
#     --------
#     I want you to find the first number (in digit or words) in the input question, and only ouput the number you found.
#     Pay attention: you must only output the number same in the origin (like <six> or <6>) in a single number or word.
#     Do not output anything else like code.

#     Input : "{problem}"
#     Output: 
#     '''
#     #pprint(prompt)
#     result = get_response(
#         prompt=prompt
#     )

#     #pprint(result)
#     question=problem.replace(result,"X",1)
#     return question,result

def construct_verify_problem(a,b):
    prompt=f'''
    I want you to decide whether two inputs are the same number, whether they are in words or in digit.
    
    Here are three examples:
    --------
    Input : "three" and "3"
    Output: True

    Input : "three" and "five"
    Output: False

    Input: "3" and "3"
    Ouput: True

    Input : "three" and "5"
    Output: False

    --------
    Pay attention: you must only output True or False. You must not output anything else like code.
    Input : "{a}" and "{b}"
    Output:
    '''

    result = get_response(
        prompt=prompt
    )

    return result


def get_response(prompt):
    # completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": f"{prompt}"}], max_tokens=max_length, temperature=0.7)
    # response = completion.choices[0].message.content    
    
    chat=pipeline([[{"role": "user", "content":f"{ prompt }"}]])
    #response=chat[0].text
    response=chat[0][0]["generated_text"][-1]["content"]

    return response

problem="The serum will make the drinker grow an extra arm every five days and an extra leg every five days. After fifteen days, how many new limbs will Helena’s serum cause a person to grow if they drink it ?"


boo=construct_verify_problem("5", "three")
pprint(f"question is <{boo}>")
#pprint(f"number is <{bar}>")


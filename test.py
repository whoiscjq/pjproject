import transformers
import torch

model_id = "models/Meta-Llama-3.1-8B-Instruct"

# 加载模型
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# 定义批量输入的prompts
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
    {"role": "user", "content": "What is your favorite treasure?"},
    {"role": "user", "content": "Tell me a pirate joke!"},
]

# 生成输出
outputs = []
for message in messages:
    input_text = message["content"]
    output = pipeline(input_text, max_new_tokens=256)
    outputs.append(output[0]["generated_text"])

# 打印所有的输出
for i, output in enumerate(outputs):
    print(f"Prompt {i+1}:")
    print(output)
    print("\n")


# import json

# # 加载JSON文件
# with open('eval.jsonl', 'r') as file:
#     data = json.load(file)

# # 打印每个问题及其答案
# count =0 
# for item in data:
#     count+=1
#     if count > 11: 
#         break
#     question = item['question']
#     answer = item['answer']
#     print("Question:", question)
#     print("Answer:", answer)
#     print("\n")

# print(outputs[0]["generated_text"][-1])
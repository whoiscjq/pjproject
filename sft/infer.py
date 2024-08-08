import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载经过SFT后的模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/internlm2-chat-1_8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("models/internlm2-chat-1_8b", trust_remote_code=True)

sft_model_path = "/mnt/workspace/llm/out/sft/best.pth"
model.load_state_dict(torch.load(sft_model_path, map_location=device))
model.to(device)

# 定义一个推理函数
def generate_response(question, max_length=512):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    outputs = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=92542
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 示例问题
questions = [
"Kelly has 5 quarters and 2 dimes. If she buys a can of pop for 55 cents, how many cents will she have left?",
"Judy teaches 5 dance classes, every day, on the weekdays and 8 classes on Saturday.  If each class has 15 students and she charges $15.00 per student, how much money does she make in 1 week?"
]

# 生成并打印答案
for question in questions:
    response = generate_response(question)
    print(f"Question: {question}")
    print(f"Response: {response}\n")

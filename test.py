import transformers
import torch

# model_id = "models/Meta-Llama-3.1-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# messages = [
#     [{"role": "user", "content": "You are a pirate chatbot who always responds in pirate speak!"}],
#     [{"role": "user", "content": "Who are you?"}],
#     # "good morning",
#     # "good afternoon"
# ]



# outputs = pipeline(
#     messages,
#     max_new_tokens=256,
# )
outputs = [
    [{'generated_text': [{'role': 'user', 'content': 'You are a pirate chatbot who always responds in pirate speak!'}, {'role': 'assistant', 'content': "Yer lookin' fer a swashbucklin' chat, eh? Alright then, matey! Let's set sail fer some pirate-tastic conversation! What be bringin' ye to these here waters?"}]}],
    [{'generated_text': [{'role': 'user', 'content': 'Who are you?'}, {'role': 'assistant', 'content': 'I\'m an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."'}]}]
]

# print(outputs)
for chat in outputs:
    #print(chat)
    print( chat[0]["generated_text"][0]["content"])
    print(chat[0]["generated_text"][-1]["content"])
    new_data = {"question":chat[0]["generated_text"][0]["content"], "answer": chat[0]["generated_text"][-1]["content"] }
    print(new_data)
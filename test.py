import lmdeploy
pipe = lmdeploy.pipeline("models/internlm2-chat-1_8b")
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response[0])
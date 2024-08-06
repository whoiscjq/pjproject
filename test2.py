import pandas as pd
import json

# 定义函数以逐行读取JSONL文件并转换为DataFrame
def jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

# 示例：将data.jsonl转换为DataFrame
file_path = 'dataset/eval.jsonl'
df = jsonl_to_dataframe(file_path)

# 打印DataFrame的前几行
print(df.head())

import random
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler


def jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df

class SFTDataset(Dataset):
    def __init__(self,df,tokenizer
                 ,max_length=514
                 ,prompt_max_len=256
                 ,answer_max_len=256):
        super().__init__()
        self.df=df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        #
        self.tokenizer = tokenizer
        self.bos=self.tokenizer.add_bos_token
        self.eos=self.tokenizer.add_eos_token
        self.pad=self.tokenizer.add_eos_token#self.tokenizer.special_tokens['<pad>']
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        prompt = self.tokenizer.encode(sample['question'],add_special_tokens=False)
        answer = self.tokenizer.encode(sample['answer'],add_special_tokens=False)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len-2]
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len-2]
        #
        input_id=prompt+[self.bos]+answer+[self.eos]
        context_length = len(prompt)
        mask_position = context_length - 1
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        if pad_len==0:
            loss_mask = [0]*context_length+[1]*(len(input_id[mask_position+1:])) + [0]*pad_len
        else:
            loss_mask = [0]*context_length+[1]*(len(input_id[mask_position+1:-pad_len])) + [0]*pad_len
        #
        input_id=np.array(input_id)
        X=np.array(input_id[:-1]).astype(np.int64)
        Y=np.array(input_id[1:]).astype(np.int64)
        loss_mask=np.array(loss_mask[:-1])
        #
        return torch.from_numpy(X),torch.from_numpy(Y),torch.from_numpy(loss_mask)
#
if __name__=="__main__":
    df=jsonl_to_dataframe("dataset/eval.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("models/internlm2-chat-1_8b_simple_sft", trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained("models/internlm2-chat-1_8b_simple_sft", trust_remote_code=True)
    train_ds = SFTDataset(df,tokenizer,max_length=512)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    for i, (X, Y,loss_mask) in enumerate(train_loader):
        print(X.shape,Y.shape)
        print(X[0])
        print(Y[0])
        print(loss_mask[0])
        break
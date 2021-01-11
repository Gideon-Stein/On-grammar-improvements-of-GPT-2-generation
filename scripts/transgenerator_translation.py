# Set the GPU for training
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import pickle
import copy
import operator
import json
import numpy as np
import pandas as pd
from functools import reduce
import argparse


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, modeling_utils, GPT2Config, modeling_gpt2, GPT2Model, GPT2PreTrainedModel, GPT2Config


def generation(model,tokenizer,condition):
    sentence = condition
    inp = torch.tensor(tokenizer.encode(condition)).unsqueeze(0)
    inp = inp.to("cuda")
    with torch.no_grad():
        for x in range(1024 - len(inp[0])): ## stop generation on the max length
            outputs = model(inp)
            predictions = outputs[0]
            new = torch.tensor([[torch.argmax(predictions[0, -1, :]).item()]])
            new = new.to("cuda")
            inp = torch.cat((inp,new),1)
            inp.to("cuda")
            if new[0][0].item() == 50256: #EOS token
                break 
        predicted_text = tokenizer.decode(inp.tolist()[0][len(tokenizer.encode(condition)):])
    return predicted_text


#For new format
def split_train_2(train_data):
    continueSet = []
    for x in range(len(train_data)):
        train_data[x] = train_data[x].split("====")
        if len(train_data[x]) != 3:
            pass
        else:
            continueSet.append(train_data[x][0] + "====" + train_data[x][1] + "====")
    return continueSet

def split_train(train_data):
    for x in range(len(train_data)):
        train_data[x] = train_data[x].split("====")
        train_data[x][0] = train_data[x][0] + "===="
    return train_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", default="data_files/test.txt", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test_data",default=False,  type = bool)
    parser.add_argument("--n_data",default=50, type = int)
    parser.add_argument("--save_path",default="data_files/test.p", type = str)
    parser.add_argument("--n_layers",default=12,type = int)
    args = parser.parse_args()
    
    if args.n_layers == 12:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.load_state_dict(torch.load(args.model_path))
    else:
        config = GPT2Config(n_layer = args.n_layers)
        model = GPT2LMHeadModel(config)
        model.load_state_dict(torch.load(args.model_path))
    model.to("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    
    if args.test_data: 
        data = pickle.load(open(args.text_path,"rb"))
        if len(data[0]) < args.n_data:
            args.n_data = len(data)
    else:
        data  = open(args.text_path,"r+",encoding="utf-8")
        data = data.read()
        data = data.split("<|endoftext|>")
        data = split_train_2(data)
        if len(data) < args.n_data:
            args.n_data = len(data)
        
    predictions = []
    if args.test_data :
        for x in range(args.n_data):
            pred = generation(model,tokenizer,data[0][x])
            predictions.append(pred)
            print(x)

    else:   
        for x in range(args.n_data):
            print(data[x])
            pred = generation(model,tokenizer,data[x])
            predictions.append(pred)
            print(x)

    pickle.dump(predictions,open(args.save_path, "wb"))



if __name__ == '__main__':
    main()
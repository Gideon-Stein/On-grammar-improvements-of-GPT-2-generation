import torch
import pickle
#from transformers import GPT2Tokenizer, GPT2LMHeadModel, modeling_utils, GPT2Config, modeling_gpt2, GPT2Model, GPT2PreTrainedModel, GPT2Config
import copy
import operator
import json
import numpy as np
import pandas as pd
import language_check
import os
from os import listdir
from os.path import isfile, join
#import language_check
from functools import reduce
import difflib
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer


# functions


def add_correct_tokens(corpus,indices,maximum):
    out = []
    counter = 0
    indices.append(maximum+5) # so list doesnt get empty
    for x in range(maximum):
 
        if indices[0] == x:
            out.append([""])
            del indices[0]
        else:
            out.append(corpus[counter])
            counter+=1    
    return out
#-------------------------------------------------------------------------

def build_dict(items):
    out = {}
    for x in items:
        if x in out: 
            out[x] +=1
        else:
            out[x] = 1
    return out

def build_dict_mistakes(items):
    out1 = {}
    out2 = {}
    for x in items:
        if str(x[:2]) in out1: 
            out1[str(x[:2])] +=1
        else:
            out1[str(x[:2])] = 1
            
        if x[2] in out2: 
            out2[x[2]] +=1
        else:
            out2[x[2]] = 1
    return out1,out2

def build_frequency_stats(data):
    rules = build_dict(data[2])
    sorted_rules  = sorted(rules.items(), key=operator.itemgetter(1))
    sorted_rules.reverse()
    types = build_dict(data[3])
    sorted_types  = sorted(types.items(), key=operator.itemgetter(1))
    sorted_types.reverse()
    specific_mistakes, sentenceErrorRate = build_dict_mistakes(data[4])
    sorted_specific_mistakes  = sorted(specific_mistakes.items(), key=operator.itemgetter(1))
    sorted_specific_mistakes.reverse()
    sorted_sentenceErrorRate  = sorted(sentenceErrorRate.items(), key=operator.itemgetter(1))
    sorted_sentenceErrorRate.reverse()
    return [sorted_rules,sorted_types,sorted_specific_mistakes,sorted_sentenceErrorRate]
#-------------------------------------------------------------------------
    
def calculate_model(model,data):
    calculate_stats(model["3"]['test_all_wrong_700'],data["test_700"],"3_test_all_wrong_700")
    calculate_stats(model["3"	]['test_all_wrong_long'],data["test_l"],"3_test_all_wrong_long")
    calculate_stats(model["3"]['test_no_filter_long'],data["test_nf_l"],"3_test_no_filter_long")
    calculate_stats(model["3"]['test_no_filter_700'],data["test_nf_700"],"3_test_no_filter_700")
    calculate_stats(model["3"]['train'],data["train"],"3_train")
    
    calculate_stats(model["6"]['test_all_wrong_700'],data["test_700"],"6_test_all_wrong_700")
    calculate_stats(model["6"]['test_all_wrong_long'],data["test_l"],"6_test_all_wrong_long")
    calculate_stats(model["6"]['test_no_filter_long'],data["test_nf_l"],"6_test_no_filter_long")
    calculate_stats(model["6"]['test_no_filter_700'],data["test_nf_700"],"6_test_no_filter_700")
    calculate_stats(model["6"]['train'],data["train"],"6_train")
    
    calculate_stats(model["full"]['test_all_wrong_700'],data["test_700"],"full_test_all_wrong_700")
    calculate_stats(model["full"]['test_all_wrong_long'],data["test_l"],"full_test_all_wrong_long")
    calculate_stats(model["full"]['test_no_filter_long'],data["test_nf_l"],"full_test_no_filter_long")
    calculate_stats(model["full"]['test_no_filter_700'],data["test_nf_700"],"full_test_no_filter_700")
    calculate_stats(model["full"]['train'],data["train"],"full_train")

def calculate_stats(translations,data,name):
    results = []
    for x in range(len(translations)):
        results.append(translation_accuracy_new_for(translations[x],data))
    pickle.dump(results, open("saves/" + name + ".p","wb"))

def correct(text,trashFilter = False):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokensN = 0
    corrected = []
    wrongN = 0
    filterCount = 0
    mistakesN = 0
    rulesApplied = []
    replacements = []
    types = []
    noMistakes = []
    sentenceN = 0
    tool = language_check.LanguageTool('en-US')
    for instance in text:
        sentence = instance.replace("<|endoftext|>","")
        if len(sentence) > 0:
            if sentence[0] == " ":
                sentence = sentence[1:]
        matches = tool.check(sentence)
        if len(matches) > 0:
            if len(matches) > 100 and trashFilter == True:
                filterCount += 1
            else:
                tokensN += len(tokenizer.encode(sentence))
                corrected.append(language_check.correct(sentence, matches))
                wrongN += 1
                for rule in matches: 
                    mistakesN +=1
                    rulesApplied.append(rule.ruleId)
                    types.append(rule.category)
                    new = rule.replacements
                    old = sentence[rule.fromx:rule.tox]
                    replacements.append((old,new,sentenceN))
                sentenceN+=1
        else:
            noMistakes.append(sentenceN)
            corrected.append(sentence)
            tokensN += len(tokenizer.encode(sentence))
            sentenceN+=1
    stats = [wrongN,mistakesN,rulesApplied,types,replacements,noMistakes,tokensN,filterCount]
    return corrected, stats

def correct_2(text,trashFilter = False):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokensN = 0
    corrected = []
    wrongN = 0
    filterCount = 0
    mistakesN = 0
    rulesApplied = []
    replacements = []
    types = []
    noMistakes = []
    sentenceN = 0
    tool = language_check.LanguageTool('en-US')
    for instance in text:
        sentence = instance.replace("<|endoftext|>","")
        if len(sentence) > 0:
            if sentence[0] == " ":
                sentence = sentence[1:]
        matches = tool.check(sentence)
        tokensN += len(tokenizer.encode(sentence))
        if len(matches) > 0:
            if len(matches) > 100 and trashFilter == True:
                filterCount += 1
            corrected.append(language_check.correct(sentence, matches))
            wrongN += 1
            for rule in matches: 
                mistakesN +=1
                rulesApplied.append(rule.ruleId)
                types.append(rule.category)
                new = rule.replacements
                old = sentence[rule.fromx:rule.tox]
                replacements.append((old,new,sentenceN))
            sentenceN+=1
        else:
            noMistakes.append(sentenceN)
            corrected.append(sentence)
            sentenceN+=1
    stats = [wrongN,mistakesN,rulesApplied,types,replacements,noMistakes,tokensN,filterCount]
    return corrected, stats

def correct_base():
    out = get_untrained_model_sent()
    corrected, stats = correct(out,True)
    pickle.dump(corrected, open("saves/base_translation.p","wb"))
    pickle.dump(stats, open("saves/base_translation_stats.p","wb"))

def correct_base_test():
    texts = open("saves/splitOnEosDataset_v2_test.txt", "r+",encoding = "UTF-8")
    texts = texts.read()
    texts = texts.split("<|endoftext|>")
    corrected, stats = correct(texts,True)
    pickle.dump(corrected, open("saves/base_translation_test.p","wb"))
    pickle.dump(stats, open("saves/base_translation_stats_test.p","wb"))

def correct_data(trans):
    for x in ["3","6", "full"]:
        for y in ['train','test_all_wrong_700','test_all_wrong_long','test_no_filter_long','test_no_filter_700']:
            print(x + y)
            model = trans[x][y]
            name = x + "_" + y
            correctStack = []
            statsStack = []
            for z in model:
                corrected, stats = correct_2(z,True)
                correctStack.append(corrected)
                statsStack.append(stats) 
            pickle.dump(correctStack, open("saves/"+ name + "_translation.p","wb"))
            pickle.dump(statsStack, open("saves/"+ name + "_translation_stats.p","wb"))

def correct_model_dataset(model,name):
    correctStack = []
    statsStack = []
    for x in model:
        corrected, stats = correct(x,True)
        correctStack.append(corrected)
        statsStack.append(stats) 
    pickle.dump(correctStack, open("saves/"+ name + "_translation.p","wb"))
    pickle.dump(statsStack, open("saves/"+ name + "_translation_stats.p","wb"))

#-------------------------------------------------------------------------

def extend_x(data):
    out = []
    for x in range(len(data)):
        out.append([data[x],[5,10,17,20]])
    return np.array(out)

def extend_x_2(data,where):
    axis = [5,10,17,20]
    out = []
    for x in data[:,:,where]: 
        out.append(np.array([x,axis]))
    return out

def extend_x_3(data):
    axis = [5,10,17,20]
    return np.array([data,axis])	
#-------------------------------------------------------------------------

def filter_trash (sentences,indices,maximum):
    out1 = []
    out2 = []
    toDelete = []
    for x in (indices):
        if x[1] > maximum:
            toDelete.append(x[0])
            
    for index in sorted(toDelete, reverse=True):
        del sentences[0][index]
        del sentences[1][index]

def filter_trash_2 (sentences,indices,maximum):
    out1 = []
    out2 = []
    toDelete = []
    for x in (indices):
        if x[1] > maximum:
            toDelete.append(x[0])
            
    for index in sorted(toDelete, reverse=True):
        del sentences[0][index]
        del sentences[1][index]
    
    print (str(len(toDelete)) + " were deleted since they had more than" +  str(maximum) + " mistakes")
    final1 = []
    final2 = []
    count = 0
    for x in range(len(sentences[1])): 
        if len(sentences[1][x]) > 1:
            final1.append(sentences[0][x])
            final2.append(sentences[1][x])
        else:
            count+=1
    print(str(count) + " sentences had no grammar mistakes. They were deleted from the dataset") 
    return [final1,final2]


def filter_trash_3 (sentences,indices,maximum):
    out1 = []
    out2 = []
    toDelete = []
    for x in (indices):
        if x[1] > maximum:
            toDelete.append(x[0])
            
    for index in sorted(toDelete, reverse=True):
        del sentences[0][index]
        del sentences[1][index]
    
    print (str(len(toDelete)) + " were deleted since they had more than" +  str(maximum) + " mistakes")
    final1 = []
    final2 = []
    count = 0
    for x in range(len(sentences[1])): 
        if len(sentences[1][x]) > 1:
            final1.append(sentences[0][x])
            final2.append(sentences[1][x])
        else:
            count+=1
            final1.append(sentences[0][x])
            final2.append(sentences[0][x] + "<|endoftext|>")
    print(str(count) + " sentences had no grammar mistakes.") 
    return [final1,final2]

def filter_trash_4(sentences,indices,maximum): #conflict class filter_trash from merge. Not used
    out = []
    toDelete = []
    for x in indices:
        if x[1] > maximum:
            toDelete.append(x[0])
    for x in range(len(sentences)):
        if x in toDelete:
            pass
        else:
            out.append(sentences[x])
    print (str(len(toDelete)) + " were deleted.")
    final = []
    for x in out: 
        if len(x) > 1:
            final.append(x)
    print(str(len(final)) + " were correct.") 
    return final
    
    print (str(len(toDelete)) + " were deleted since they had more than" +  str(maximum) + " mistakes")
    return sentences


def finetuned(path):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load("transformers/examples/model_save/" + str(path)))
    return model

def format_reloads_dataset(data1,data2,data3,key):
    titles = ["Correct Trans","Trans included", "Rep included","Real in cond","Len Trans = Real","Length Trans","Length Real","Average Levenstein Real/Trans"]
    plt.figure(figsize=(20,20))
    formated1 = extend_x(np.array(data1[key]).T)
    formated3 = extend_x(np.array(data3[key]).T)
    for x in range(len(titles)):
        plt.subplot(int(str(42) + str(x+1)))
        plt.title(titles[x])
        plt.plot(formated1[x][1],formated1[x][0])
        plt.plot(np.array(data2[key]).T[x])
        plt.plot(formated3[x][1],formated3[x][0])
        plt.legend(["l3","l6",'full'])

def format_reloads_dataset_2(data,key):
    titles = ["Number of sentences", "Number of wrong sentences", "Number of Mistakes", "Number of tokens",
          "Errors per Token","Error per sentence"]
    plt.figure(figsize=(20,20))
    values = [data["3"][key],data["6"][key],data["full"][key]]
    for x in range(len(titles)):
        plt.subplot(int(str(42) + str(x+1)))
        plt.title(titles[x])
        plt.plot([5,10,17,20],np.array(values[0]).T[x].tolist())
        plt.plot(np.array(values[1]).T[x])
        plt.plot([5,10,17,20],np.array(values[2]).T[x].tolist())
        plt.legend(["l3","l6",'full'])

def format_reloads_model(data):
    titles = ["Correct Trans","Trans included", "Rep included","Real in cond","Len Trans = Real","Length Trans","Length Real","Average Levenstein Real/Trans"]
    train = np.array(data["train"]).T.tolist()
    test1 = np.array(data["test_700"]).T.tolist()
    test2 = np.array(data['test_long']).T.tolist()
    test3 = np.array(data['test_nf_long']).T.tolist()
    test4 = np.array(data['test_nf_700']).T.tolist()
    graphs = {}
    for x in range(len(train)):
        graphs[titles[x]] = np.array([train[x],test1[x],test2[x],test3[x],test4[x]])
    plt.figure(figsize=(20,20))
    for x in range(len(titles)):
        plt.subplot(int(str(42) + str(x+1)))
        plt.title(titles[x])
        if len(graphs[titles[x]].T) != 20: 
            formated = extend_x(graphs[titles[x]])
            for w in formated:
                plt.plot(w[1],w[0])
        else:
            plt.plot(graphs[titles[x]].T)
        plt.legend(["train","test_700",'test_long','test_nf_long','test_nf_700'])    

def format_reloads_model_2(data,key):
    titles = ["Number of sentences", "Number of wrong sentences", "Number of Mistakes", "Number of tokens",
          "Errors per Token","Error per sentence","trash filtered"]
    plt.figure(figsize=(20,20))
    
    for x in range(len(titles)):
        plt.subplot(int(str(42) + str(x+1)))
        plt.title(titles[x])
        if key != "6":
            formatted = extend_x_2(data[key],x)
            for graph in formatted: 
                plt.plot(graph[1],graph[0])      
        else:
            plt.plot(data[key][:,:,x].T)
        plt.legend(["train","test_700",'test_long','test_nf_long','test_nf_700'])

def format_reloads_model_3(translation,finetune):
    titles = ["Errors per Token", "Errors per sentence"]
    plt.figure(figsize=(20,20))
    
    for x in range(len(titles)):
        plt.subplot(int(str(12) + str(x+1)))
        plt.title(titles[x])
        formatted = extend_x_3(translation["3"][3][:,x+4])
        plt.plot(formatted[1],formatted[0]) 
        formatted = extend_x_3(translation["full"][3][:,x+4])
        plt.plot(formatted[1],formatted[0])   
        plt.plot(translation["6"][x][:,x+4])  
        plt.plot(finetune[:,x+4])
        plt.legend(["3Layer","12 Layer",'6Layer','Classic Finetune'])


def format_sentence_2(sentence):
    sentence = sentence.replace("<|endoftext|>","")
    if len(sentence) > 0:
        while sentence[0] == " " or sentence[0] == "\n":
            sentence = sentence[1:]
            if len(sentence) == 0:
                break
    if len(sentence) > 0:
        while sentence[-1] == " "  or sentence[-1] == "\n":
            sentence = sentence[:-1]
            if len(sentence) == 0:
                break
    return sentence
#-------------------------------------------------------------------------            

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

def get_untrained_model_sent(maxN = None):
    texts = []
    for i, line in enumerate(open("original_data/small-117M." + "train" +".jsonl")):
        texts.append(json.loads(line)['text'])
    if maxN == None:
        return texts
    else: 
   	    return texts[:maxN]

def grammar_stats(stats,nsentences=1000):
    nWrong = stats[0]
    nMistakes = stats[1]
    ept = (nMistakes/stats[-2])
    eps = (stats[1]/nsentences)
    out = [nsentences,nWrong,nMistakes,stats[-2],ept,eps,stats[-1]]
    return out 
#-------------------------------------------------------------------------            

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def load_and_split_finetune():
    onlyfiles = [f for f in listdir("generated/classic") if isfile(join("generated/classic", f))]
    out = [None]* len(onlyfiles)
    for file in onlyfiles:
        where = int(file.split("_")[-1][:-4])
        texts = open("generated/classic/" + file,"r+",encoding="utf-8")
        texts = texts.read()
        texts = texts.split("<|endoftext|>")
        prep = []
        for block in texts:
            sentence = block.replace(". ", ".<|splitter|>")
            sentence = sentence.replace("? ", "?<|splitter|>")
            sentence = sentence.replace("! ", "!<|splitter|>")
            sentence = sentence.replace(".\n", ".\n<|splitter|>")
            sentence = sentence.replace(".\n\n", ".\n\n<|splitter|>")
            sentence = sentence.replace("?\n", "?\n<|splitter|>")
            sentence = sentence.replace("?\n\n", "?\n\n<|splitter|>")
            sentence = sentence.replace("!\n", "!\n<|splitter|>")
            sentence = sentence.replace("!\n\n", "!\n\n<|splitter|>")
            prep.append(sentence)
        splitted = []
        for block in prep: 
            splitted.append(block.split("<|splitter|>"))
        final = [item for sublist in splitted for item in sublist]
        out[where-1] = final[:1000]
    return out

def load_base():
    stats = pickle.load(open("saves/base_translation_stats_test.p","rb"))
    out = grammar_stats(stats,len(stats[-3]) + stats[0])
    return out

def load_data():
    out  ={}
    out["3"] = load_model("generated/3layer")
    out["6"] = load_model("generated/6layer")
    out["full"] = load_model("generated/full")
    return out

def load_datasets():
    train = open("build_data/EOS_new_full_train_5K.txt","r+",encoding="utf-8")
    train = train.read()
    train = train.split("<|endoftext|>")
    train = split_train_2(train)
    
    test1 = open("build_data/EOS_new_filter_long.txt","r+",encoding="utf-8")
    test1 =  test1.read()
    test1 =  test1.split("<|endoftext|>")
    test1 = split_train_2( test1)
    
    test2 = open("build_data/EOS_new_filter_700.txt","r+",encoding="utf-8") # needs rerun no data available
    test2 =  test2.read()
    test2 =  test2.split("<|endoftext|>")
    test2 = split_train_2( test2)
    
    test3 = open("build_data/EOS_new_no_filter_long.txt","r+",encoding="utf-8")
    test3 =  test3.read()
    test3 =  test3.split("<|endoftext|>")
    test3 = split_train_2( test3)
    
    test4 = open("build_data/EOS_new_no_filter_700.txt","r+",encoding="utf-8")
    test4 =  test4.read()
    test4 =  test4.split("<|endoftext|>")
    test4 = split_train_2( test4)
    
    out = {}
    out["train"] = train
    out["test_l"] = test1
    out["test_700"] = test2
    out["test_nf_l"] = test3 
    out["test_nf_700"] = test4
    return out

def load_examples(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    out = [None] * len(onlyfiles)
    for x in onlyfiles: 
        where = int(x.split("_")[-1][:-2])
        out[where-1] = pickle.load(open(folder + "/" + x, "rb"))
    return out

def load_finetune():
    out = []
    stats = pickle.load(open("saves/classic_finetuning_translation_stats.p","rb"))
    cor = pickle.load(open("saves/classic_finetuning_translation.p","rb"))
    for x in range (len(stats)):
        out.append(grammar_stats(stats[x],len(stats[x][-3]) + stats[x][0]))
    return out

def load_gpt_data():
    texts = []
    for i, line in enumerate(open("gpt2-dataset/small-117M." + str("train") +".jsonl")):
        texts.append(json.loads(line)['text'])
    inp1= pickle.load( open( "gpt2-dataset/corrected" + str("train100k") + ".p", "rb" ))
    inp2 = pickle.load( open( "gpt2-dataset/corrected" + str("train150k")+ ".p", "rb" ))
    inp3 = pickle.load( open( "gpt2-dataset/stats" + str("train100k") + ".p", "rb" ))
    inp4= pickle.load( open( "gpt2-dataset/stats"+ str("train150k") + ".p", "rb" ))
    return [texts,[inp1,inp2],[inp3,inp4]]

def load_gpt_data_2():
    texts = []
    for i, line in enumerate(open("gpt2-dataset/small-117M." + str("test") +".jsonl")):
        texts.append(json.loads(line)['text'])
    inp1= pickle.load( open( "gpt2-dataset/corrected" + str("test") + ".p", "rb" ))
    inp3 = pickle.load( open( "gpt2-dataset/stats" + str("test") + ".p", "rb" ))
    return [texts,inp1,inp3]

def load_grammar():
    out = {}
    for x in ["3","6","full"]:
        model = []
        for y in ['train','test_all_wrong_700','test_all_wrong_long','test_no_filter_long','test_no_filter_700']:
            stats = pickle.load(open("saves/"+ x + "_" + y + "_translation_stats.p","rb"))
            package = []
            for z in stats:
                package.append(grammar_stats(z,len(z[-3]) + z[0]))
            model.append(package)
        out[x] = np.array(model)  
    return out

def load_manual_data(length):
    inp1 = open("manual_dataset/gpt2_generation1.txt"+ str(length),encoding="UTF-8")
    inp1 = inp1.read()
    inp1 = inp1.split("<|endoftext|>")
    inp2 = pickle.load( open( "manual_dataset/corrected" + str(length) + ".p", "rb" ))
    inp3 = pickle.load( open( "manual_dataset/stats" + str(length) + ".p", "rb" ))
    return [inp1,inp2,inp3]

def load_model(model):
    out = {}
    output = [dI for dI in os.listdir(model) if os.path.isdir(os.path.join(model,dI))]
    for x in output: 
        if x[0] == ".":
            pass
        else:
            inp =load_examples(model + "/" + x)
            out[x] = inp
    return out
#-------------------------------------------------------------------------  

def reload(model):
    out = {}
    out["test_700"] = pickle.load(open("saves/" + model + "_test_all_wrong_700.p","rb"))
    out["test_long"] =pickle.load(open("saves/" + model + "_test_all_wrong_long.p","rb"))
    out["test_nf_long"] =pickle.load(open("saves/" + model + "_test_no_filter_long.p","rb"))
    out["test_nf_700"] =pickle.load(open("saves/" + model + "_test_no_filter_700.p","rb"))
    out["train"] =pickle.load(open("saves/" + model + "_train.p","rb"))
    return out
#-------------------------------------------------------------------------  

def split_eos(data):
    prep = []
    for block in data:
        sentence = block.replace(". ", ".<|splitter|>")
        sentence = sentence.replace("? ", "?<|splitter|>")
        sentence = sentence.replace("! ", "!<|splitter|>")
        sentence = sentence.replace(".\n", ".\n<|splitter|>")
        sentence = sentence.replace(".\n\n", ".\n\n<|splitter|>")
        sentence = sentence.replace("?\n", "?\n<|splitter|>")
        sentence = sentence.replace("?\n\n", "?\n\n<|splitter|>")
        sentence = sentence.replace("!\n", "!\n<|splitter|>")
        sentence = sentence.replace("!\n\n", "!\n\n<|splitter|>")
        prep.append(sentence)
    splitted = []
    for block in prep: 
        splitted.append(block.split("<|splitter|>"))
    return splitted

def split_train_2(train_data):
    continueSet = []
    target = []
    for x in range(len(train_data)):
        train_data[x] = train_data[x].split("====")
        if len(train_data[x]) != 3:
            pass
        else:
            continueSet.append(train_data[x][0] + "====" + train_data[x][1] + "====")
            target.append(train_data[x][2])
    return [continueSet,target]
#-------------------------------------------------------------------------  

def translation_accuracy_new_for(translations,data,maxCount=1000):
    count = 0
    somewhere = 0
    repSomewhere = 0
    noMistake = 0
    correctL = 0
    AverageLengthT = 0
    AverageLengthR = 0
    ALevenstein = 0
    for x in range (maxCount):
        translation = format_sentence_2(translations[x])
        cond = format_sentence_2(data[0][x])
        real = format_sentence_2(data[1][x])
        if real == translation: 
            count +=1
        if real in translation:
            somewhere +=1
        if translation in cond:
            repSomewhere +=1
        if real in cond:
            noMistake +=1 
        if len(translation) == len(real):
            correctL +=1
        AverageLengthT += len(translation)
        AverageLengthR += len(real)
        ALevenstein += levenshtein(translation, real)
    return [count/maxCount,somewhere/maxCount, repSomewhere/maxCount, noMistake/maxCount,
            correctL/maxCount,AverageLengthT/maxCount,AverageLengthR/maxCount,ALevenstein/maxCount]  
#-------------------------------------------------------------------------  

# classes
class filter_examples:
    def __init__(self): 
        self.trans = load_data()
        self.org = self.sort_base()
        del self.org[515]   # Bugfix (some additional sentence is notin the test translations. No idea why)
        self.cor, self.stats = correct_2(self.trans["full"]['test_no_filter_long'][3])
        self.corB, self.statsB = correct(self.org,True)
        self.mistakeDict,   self.mistakeDictB = self.get_error_types()
        self.upgr = self.get_upgrades()
        self.copies = self.get_mistake_copies()
    
    def get_error_types(self):
        types = {}
        for x in range(len(self.stats[2])):
            if self.stats[2][x] not in types: 
                types[self.stats[2][x]] = [self.stats[4][x][-1]]
            else: 
                types[self.stats[2][x]].append(self.stats[4][x][-1])
                
                
        typesB = {}
        for x in range(len(self.statsB[2])):
            if self.statsB[2][x] not in typesB:
                typesB[self.statsB[2][x]] = [self.statsB[4][x][-1]]
            else: 
                typesB[self.statsB[2][x]].append(self.statsB[4][x][-1])
        return types, typesB
    
    def sort_base(self):
        inp1 = open("saves/splitOnEosDataset_v2_test.txt",encoding="UTF-8")
        inp1 = inp1.read()
        inp1 = inp1.split("<|endoftext|>")
        inp2 = pickle.load( open( "saves/EOS_corrected_v2_test.p", "rb" ))
        inp3 = pickle.load( open( "saves/EOS_stats_v2_test .p", "rb" ))
        stats = pickle.load( open( "saves/EOS_freq_v2_test.p", "rb" ))
        inp2 = add_correct_tokens(inp2,inp3[5],len(inp1))
        sentences = (inp1,inp2)
        filtered1 = filter_trash_3(sentences,stats[-1],99)
        base = filtered1[0][:2002]
        filtered = []
        for x in range(len(base)):
            if x %2 == 1:
                filtered.append(base[x])
        return filtered
    
    def get_mistake_types(self):
        print("TRANSLATIONS:" + str(self.mistakeDict.keys()))
        print("Original:" + str(self.mistakeDictB.keys()))
    
    def get_upgrades(self):
        upgrades = {}
        for key in self.mistakeDictB: 
            for x in self.mistakeDictB[key]:
                if key in self.mistakeDict.keys():
                    if x not in self.mistakeDict[key]:
                        if key not in upgrades:
                            upgrades[key] = [x]
                        else: 
                            upgrades[key].append(x)
                else: 
                    upgrades[key] = [self.mistakeDictB[key], "ALL upgraded"]
        return upgrades


    def get_mistake_copies(self):
        copies = {}
        for key in self.mistakeDictB: 
            for x in self.mistakeDictB[key]:
                if key in self.mistakeDict.keys():
                    if x in self.mistakeDict[key]:
                        if key not in copies:
                            copies[key] = [x]
                        else: 
                            copies[key].append(x)
        return copies
    
    def upgrade_example(self,Mtype,nr=1):
        if self.upgr[Mtype][-1] =="ALL upgraded":
            print("Mistake type: " + Mtype + " /ALL upgraded") 
            nrs = self.upgr[Mtype][0][:nr]
        else:
            print("Mistake type: " + Mtype)  
            nrs = self.upgr[Mtype][:nr]
        for y in nrs: 
            print("Original:" + self.org[y])
            print("Translation:"  + self.trans["full"]['test_no_filter_long'][0][y])
            print(y)

    def copy_example(self,Mtype,nr=1):
        if Mtype not in self.copies.keys():
            print("No copies of this error found")
        else:
            print("Mistake type: " + Mtype)  
            nrs = self.copies[Mtype][:nr]
        for y in nrs: 
            print("Original:" + self.org[y])
            print("Translation:"  + self.trans["full"]['test_no_filter_long'][0][y])
            print("Correct:" + self.corB[y] )

            print(y)
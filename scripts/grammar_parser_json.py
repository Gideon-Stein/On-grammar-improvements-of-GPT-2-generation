import json
import numpy as np
import language_check
import argparse
import glob
import pickle

tool = language_check.LanguageTool('en-US')

def load(path,filter):
    texts = []
    for i, line in enumerate(open(path)):
        texts.append(json.loads(line)['text'])
    if filter:
        texts =  filter_symbols(texts)
    return texts

def correct(text,reps):
    corrected = []
    wrongN = 0
    sentenceN = 0
    mistakesN = 0
    rulesApplied = []
    replacements = []
    types = []
    noMistakes = []
    tool = language_check.LanguageTool('en-US')
    for sentence in text: 
        matches = tool.check(sentence)
        if len(matches) > 0: 
            corrected.append(language_check.correct(sentence, matches))
            wrongN += 1
            for rule in matches: 
                mistakesN +=1
                rulesApplied.append(rule.ruleId)
                types.append(rule.category)
                new = rule.replacements
                old = sentence[rule.fromx:rule.tox]
                if reps == True:
                    replacements.append((old,new,sentenceN))
        else:
        	noMistakes.append(sentenceN)

        if sentenceN%100 == 0:
            print("sentence " + str(sentenceN)+ "was corrected")
        sentenceN += 1
    stats = [wrongN,mistakesN,rulesApplied,types,replacements,noMistakes]
    return corrected, stats


def main():   # split corpus here win index manually do two smaller files (run script twice)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="test.txt", type=str)
    parser.add_argument("--save_replace", default=False, type=bool)
    parser.add_argument("--name", default="NoName", type=str)
    parser.add_argument("--filter", default=False, type=bool)
    args = parser.parse_args()
    
    corpus = load(args.path,args.filter)
    autocorrect, stats = correct(corpus, args.save_replace)



    for x in range(len(autocorrect)):
    	autocorrect[x] = autocorrect[x] +"<|endoftext|>" 


    pickle.dump(autocorrect,open("C_only_gen/new comparison/corrected" + str(args.name) + ".p","wb"))
    pickle.dump(stats,open("C_only_gen/new comparison/stats" + str(args.name) + ".p","wb"))

   


if __name__ == '__main__':
    main()
import language_check
import argparse
import glob
import pickle
from transformers import GPT2Tokenizer

tool = language_check.LanguageTool('en-US')

def load(path):
    corpus = []
    corpus =  open(path, 'r', encoding="utf-8")
    corpus = corpus.read()
    corpus = corpus.split("<|endoftext|>")
    return corpus


def correct(text,trashFilter = False):
    report = 0
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
    print(len(text))
    for instance in text:
        report += 1
        if report % 1000 == 0:
            print(report)
        sentence = instance.replace("<|endoftext|>","")
        if len(sentence) > 0:
            if sentence[0] == " ":
                sentence = sentence[1:]
        matches = tool.check(sentence)
        if len(matches) > 0:
            if len(matches) > 100 and trashFilter == True:
                filterCount += 1
            else:
                #tokensN += len(tokenizer.encode(sentence))
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
            #tokensN += len(tokenizer.encode(sentence))
            sentenceN+=1
    stats = [wrongN,mistakesN,rulesApplied,types,replacements,noMistakes,tokensN,filterCount]
    return corrected, stats



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="test.txt", type=str)
    parser.add_argument("--save_replace", default=False, type=bool)
    parser.add_argument("--splitnr", default="None", type=str)
    args = parser.parse_args()
    
    corpus = load(args.path)
    autocorrect, stats = correct(corpus, args.save_replace)



    for x in range(len(autocorrect)):
    	autocorrect[x] = autocorrect[x] +"<|endoftext|>" 


    pickle.dump(autocorrect,open("EOS_corrected_v2_" +  args.nr + ".p","wb"))
    pickle.dump(stats,open("EOS_stats_v2_" + args.nr + ".p","wb"))
   


if __name__ == '__main__':
    main()
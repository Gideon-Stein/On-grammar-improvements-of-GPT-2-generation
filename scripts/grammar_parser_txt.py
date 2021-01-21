import language_check
import argparse
import glob
import pickle

tool = language_check.LanguageTool('en-US')

def load(path):
    corpus = []
    corpus =  open(path, 'r', encoding="utf-8")
    corpus = corpus.read()
    corpus = corpus.split("<|endoftext|>")
    return corpus


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

        if sentenceN%1000 == 0:
            print("sentence " + str(sentenceN)+ "was corrected")
        sentenceN += 1
    stats = [wrongN,mistakesN,rulesApplied,types,replacements,noMistakes]
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


    pickle.dump(autocorrect,open("EOS_corrected_v2_"  args.nr + ".p","wb"))
    pickle.dump(stats,open("EOS_stats_v2_" + args.nr + ".p","wb"))
   


if __name__ == '__main__':
    main()
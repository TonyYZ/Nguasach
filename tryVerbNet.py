import nltk
nltk.download('framenet_v17')
from nltk.corpus import framenet as fn
from lemminflect import getInflection

print(fn.frames(r'(?i)noise'))
'''
myFrame = fn.frame(39)

lst = [myFrame.name, myFrame.definition, len(myFrame.lexUnit), myFrame.FE['Voice'], myFrame.frameRelations]
for elem in lst:
    print (elem)
'''

'''
corpus = []

print(fn.lu)
with open("frameCorpus.txt", 'w', encoding='utf-8') as f:
    total = 0
    for lex in fn.lus():
        pair = lex.name.split('.')
        part_of_sp = pair[1]
        if part_of_sp != 'v':
            continue
        cur = lex.frame  # current frame
        # print(cur)
        elems = []
        for key in cur.FE.keys():
            if cur.FE[key]['coreType'] == 'Core':
                elems.append(key.lower().replace('_', ' ') + ' (' + cur.FE[key]['definition'].split('.')[0].lower().replace('<fex name="">', '') + ')')
                # elems.append(key.lower().replace('_', ' '))
        word = pair[0]
        # ger_v = getInflection(word, tag='VBG')[0].rstrip()
        # sent = "The process of " + ger_v + " involves " + ', '.join(elems) + '.'
        sent = word + ": " + ', '.join(elems)
        print(sent)
        total += 1
        f.write(sent + '\n')

print("Finished generating corpus! Total:", total)
'''

for cur in fn.frames():
    elems = []
    for key in cur.FE.keys():
        if cur.FE[key]['coreType'] == 'Core':
            elems.append(key.lower().replace('_', ' '))
    print(cur.name + ':', ', '.join(elems))
import pickle

print("Loading previously saved dictionaries...")
with open("uniMap.pk", 'rb') as fi:
    uniMap = pickle.load(fi)

def invDict(map):
    invMap = {}
    for k, v in map.items():
        invMap[v] = invMap.get(v, []) + [k]
    return invMap

invUniMap = invDict(uniMap)


print(len(invUniMap.keys()))
hist = {k: 0 for k in uniMap.values()}  # a histogram that records the frequencies of each semantic word

hexDoubleNames = {'h': '坤卦靜態', 'g': '艮卦靜態', 'r': '坎卦靜態', 'v': '巽卦靜態', 'n': '中卦靜態', 'd': '震卦靜態', 'b': '離卦靜態', 's': '兌卦靜態', 'ṡ': '乾卦靜態',
'ċ': '坤卦動態', 'c': '艮卦動態', 'l': '坎卦動態', 'f': '巽卦動態', 'm': '中卦動態', 't': '震卦動態', 'p': '離卦動態', 'z': '兌卦動態', 'ż': '乾卦動態'}

hexNames = {'h': '坤卦', 'g': '艮卦', 'r': '坎卦', 'v': '巽卦', 'n': '中卦', 'd': '震卦', 'b': '離卦', 's': '兌卦', 'ṡ': '乾卦'}

invHexDoubleNames = {k: v[0] for k, v in invDict(hexDoubleNames).items()}

invHexNames = {k: v[0] for k, v in invDict(hexNames).items()}
invHexNames.update(invHexDoubleNames)
print(invHexNames)
targetLst = ['ht', 'ċb']  # a list of hex combinations where we want to find the thickest part of their overlapping
singleMode = True  # a semantic word is only counted once for every hexagram combination
doubleMode = True  # use double hex or single hex

def trans2DoubleHex(str):
    return ''.join([hexDoubleNames[c] for c in str])

def trans2Hex(str):
    return ''.join([hexNames[c] for c in str])

def trans2Char(hexCombo):
    return ''.join([invHexNames[hex] for hex in hexCombo.split()])

def updateHist(target, hist):
    if doubleMode:
        hexSeq = [trans2DoubleHex(str) for str in target]
        targetHex = [' '.join(hexSeq), ' '.join(hexSeq[::-1])]
    else:
        targetHex = [trans2Hex(target)]
    with open("Uni" + "Double" * doubleMode + "Hex.txt", encoding='utf-8') as f:
        content = f.readlines()
        # print(content[2])
        for i, line in enumerate(content):
            phase = i % 3
            if phase == 0:
                recordMode = line.strip() in targetHex  # the subsequent words are used for the update when it is on
            elif phase == 1:
                continue
            elif i % 3 == 2 and recordMode:
                wordLst = line.split()
                if singleMode:
                    addedWords = []
                    for w in wordLst:
                        if uniMap[w] not in addedWords:
                            hist[uniMap[w]] += 1
                            addedWords.append(uniMap[w])
                else:
                    for w in wordLst:
                        hist[uniMap[w]] += 1

def getConcise(sem, soundMode=False):
    langWords = invUniMap[sem]  # words in different languages referring to the same meaning
    rangeMap = {}  # maps from a hexagram to a semantic range size
    with open("Uni" + "Double" * doubleMode + "Hex.txt", encoding='utf-8') as f:
        content = f.readlines()
        # print(content[2])
        for i, line in enumerate(content):
            phase = i % 3
            if phase == 0:
                curHex = line.strip()  # current hexagram combination
            elif phase == 1:
                if not soundMode:
                    continue
                else:
                    curSound = ', '.join(line.strip().split(', ')[:5])
            elif i % 3 == 2:
                wordLst = line.split()
                for w in langWords:
                    if w in wordLst and curHex not in rangeMap:
                        if not soundMode:
                            rangeMap[(curHex, w)] = len(wordLst)
                        else:
                            rangeMap[(curHex, curSound, w)] = len(wordLst)
    return rangeMap

def getRanking(origHist, num, needMax=True, verbose=False):
    hist = origHist.copy()
    rank = []
    for i in range(num):
        extInd = max(hist, key=hist.get) if needMax else min(hist, key=hist.get)
        if isinstance(extInd, tuple):
            caption = ' '.join(extInd) if verbose else extInd[0]
        else:
            caption = extInd
        rank.append(caption + (' (' + str(hist[extInd]) + ')') * verbose)
        hist[extInd] = 0 if needMax else float('inf')
    return rank

def main():
    needMax = False
    semRanges = getConcise('potion_', soundMode=True)
    print('hex rankings', getRanking(semRanges, 5, needMax=needMax, verbose=True))
    semRank = [trans2Char(hexCombo) for hexCombo in getRanking(semRanges, 2, needMax=needMax, verbose=False)]
    print("char rankings", semRank)
    targetLst = semRank
    for target in targetLst:
        updateHist(target, hist)
    maxPoss = [k + ' ' + str(v) for k, v in hist.items() if v == max(hist.values())]  # estimated words with the thickest degree of overlapping -> higher possibility
    print(len(maxPoss), maxPoss)
    print(getRanking(hist, 34, verbose=True))


if __name__ == "__main__":
    main()

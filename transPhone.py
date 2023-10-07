import csv
import math
import subprocess

import torch
from pyvis.network import Network
import matplotlib
import numpy as np
import random
from itertools import combinations, permutations
from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer
from sklearn.manifold import TSNE
from transvec.transformers import TranslationWordVectorizer
from mpl_toolkits import mplot3d
from scipy.special import softmax
import pickle
import statistics
from suffix_trees import STree
import neuralNetwork
import seaborn as sns
import pandas as pd
import os
import processAll
from alive_progress import alive_bar

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.style.use('ggplot')
import colorsys
import warnings

warnings.filterwarnings("ignore")

directory = "D:\\Project\\Work in Progress\\phonetic-similarity-vectors-master"
global bilingual_model
global indexer
global source_model
global target_model

initialExecution = False
useNeuralNetwork = False
useVecMap = False
useUni = False
needShuffle = True

'''
hexLabels = [['earth', 'nourish', 'mother', 'all-inclusive', 'square', 'cow', 'calf', 'soft', 'supple', 'belly', 'cloth', 'vessel', 'void', 'carriage', 'flat', 'skimpy', 'varied', 'multitude', 'handle', 'support', 'store'],
             ['mountain', 'stop', 'obstruct', 'limit', 'finish', 'footpath', 'stone', 'gravel', 'gateway', 'melon', 'guard', 'mouse', 'dog', 'finger', 'hand', 'rigid', 'hard', 'stable', 'joints'],
             ['water', 'sink', 'moist', 'abyss', 'channel', 'ditch', 'hide', 'straighten', 'bend', 'bow', 'wheel', 'anxiety', 'heart', 'toil', 'blood', 'spine', 'peril', 'through', 'moon', 'bandit', 'thorn', 'ear', 'hear', 'winter', 'north'],
             ['wind', 'permeates', 'seep', 'obedient', 'disturb', 'wood', 'rope', 'long', 'high', 'undecided', 'odor', 'scent', 'advancing', 'receding', 'income', 'leg', 'chicken'],
             # ['middle', 'average', 'same', 'equal', 'neutral', 'indifferent', 'ordinary', 'common', 'standard', 'nothing', 'universal', 'thing', 'object', 'matter', 'material', 'conserved', 'constant', 'balance', 'mind'],
             ['adapt', 'mixed', 'flexible', 'vague'],
             ['thunder', 'move', 'spread', 'develop', 'rise', 'dragon', 'highway', 'decisive', 'vehemently', 'green', 'bamboo', 'reeds', 'active', 'lush', 'vivid', 'young', 'foot', 'spring', 'east'],
             ['fire', 'bright', 'adhere', 'beautiful', 'warm', 'sun', 'lightning', 'armor', 'helmet', 'weapon', 'dry', 'shell', 'tortoise', 'crab', 'hollow', 'eyes', 'see', 'wrap', 'meet', 'summer', 'south'],
             ['marsh', 'joy', 'mouth', 'speak', 'express', 'communicate', 'witch', 'break', 'broken', 'shedding', 'sharp', 'mistress', 'sheep', 'exchange', 'autumn', 'west'],
             ['sky', 'pure', 'strong', 'firm', 'round', 'perfect', 'horse', 'head', 'father', 'war', 'king', 'ruler', 'govern', 'jade', 'metal', 'cold', 'ice', 'crimson', 'good', 'old', 'thin', 'belligerent', 'fruit'],
            ]

hexLabels = [['void', 'empty', 'static', 'motionless', 'sparse', 'few', 'clear', 'clean', 'quiet', 'silent'],
             ['return', 'retreat', 'come', 'backward', 'descend', 'cover', 'limitation', 'constrain'],
             ['occupy', 'middle', 'center', 'source', 'single', 'radiate', 'separate', 'disperse'],
             ['leak', 'seep', 'permeates', 'shed', 'erode', 'dilute', 'discard', 'depletion'],
             ['adapt', 'mixed', 'flexible', 'vague'],
             ['depart', 'advance', 'go', 'forward', 'rise', 'ascend', 'support', 'base', 'carry', 'load'],
             ['merge', 'combine', 'enclose', 'surround', 'wrap', 'juxtaposition', 'parallel', 'pair'],
             ['express', 'show', 'release', 'reveal', 'emit', 'give', 'issuing', 'deliver', 'swell'],
             ['numerous', 'tumultuous', 'energetic', 'full']
            ]
'''
hexLabels = [['void', 'empty', 'sparse', 'few', 'clean', 'spacious', 'hollow'],
             ['static', 'motionless', 'quiet', 'calm', 'fixed', 'dead'],
             ['cover', 'protect', 'high', 'top', 'roof', 'end', 'limit', 'restrict'],
             ['return', 'retreat', 'come', 'descend', 'backward', 'receive'],
             ['occupy', 'middle', 'center', 'unit', 'obstruct'],
             ['radiate', 'separate', 'scatter', 'disperse', 'spread', 'repel', 'source'],
             ['inflated', 'exaggerated', 'undermined', 'overblown', 'puffy'],
             ['leak', 'seep', 'permeates', 'shed', 'erode', 'discard'],
             ['vague', 'monotonous'],
             ['adapt', 'flexible', 'mixed'],
             ['support', 'carry', 'base', 'foundation', 'low', 'bottom', 'floor'],
             ['depart', 'advance', 'go', 'rise', 'ascend', 'forward', 'start', 'leave'],
             ['surround', 'periphery', 'adjacent', 'juxtaposition'],
             ['merge', 'combine', 'enclose', 'concentrate', 'join', 'meet', 'connect', 'attract'],
             ['facing', 'protruding', 'incomplete', 'swelling', 'stub'],
             ['express', 'release', 'reveal', 'emit', 'give', 'grow', 'discharge'],
             ['numerous', 'full'],
             ['tumultuous', 'energetic', 'lively']
             ]

# hexLabels = [['cow'], ['dog'], ['pig'], ['chicken'], ['dragon'], ['phoenix'], ['sheep'], ['horse']]
# hexNames = ["坤卦", "艮卦", "坎卦", "巽卦", "中卦", "震卦", "離卦", "兌卦", "乾卦"]
hexNames = []
for h in ["坤卦", "艮卦", "坎卦", "巽卦", "中卦", "震卦", "離卦", "兌卦", "乾卦"]:
    hexNames.append(h + "靜態")
    hexNames.append(h + "動態")
global hexVectors
global uniMap


def shuffleLang(langs):
    print("Shuffling languages...")
    temp = list(zip(*[labelDict[lang] for lang in langs]))
    random.shuffle(temp)
    for i, res in enumerate(zip(*temp)):
        labelDict[langs[i]] = list(res)
    print("Shuffling done")


if initialExecution:
    with open('nguasachV.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        labelsUnT = []
        for row in reader:  # each row is a list
            labelsUnT.append(row)
        labels = np.transpose(labelsUnT)

    labelDict = {}
    modelDict = {}
    if useUni:
        ac = []
        labelDict["Semantics"] = []
        for row in labels:
            if row[0] == 'Semantics':
                for w in row[1:]:
                    if w == '':
                        continue
                    label = w.rstrip()
                    labelDict["Semantics"] += [label]
            else:
                # if row[0] != 'Semantics' and row[0] != 'English':
                #    continue
                labelsPos = set()
                for w in row[1:]:
                    if w == '':
                        continue
                    label = w.rstrip().replace(",", "").replace(" ", "-")
                    while label in labelsPos:
                        label += "/"
                    labelsPos.add(label)
                    label += "@" + processAll.langAbb[row[0]]
                    ac += [label]
        labelDict["Uni"] = ac  # the original written form of every word
        uniMap = {}
        for i, word in enumerate(labelDict["Uni"]):
            uniMap[word] = labelDict["Semantics"][i % len(labelDict["Semantics"])]
        if needShuffle:
            shuffleLang(['Uni'])

        for lang in ["Uni", "Semantics"]:
            print(lang)
            modelDict[lang] = KeyedVectors.load_word2vec_format(directory + "\\" + lang + "Emb.txt", binary=False)
            print("Normalizing vectors...")
            modelDict[lang].init_sims(replace=True)
    else:
        for row in labels:
            # if row[0] != 'Semantics' and row[0] != 'English':
            #    continue
            ac = []
            labelsPos = set()
            for w in row[1:]:
                if w == '':
                    continue
                label = w.rstrip().replace(",", "").replace(" ", "-")
                while label in labelsPos:
                    label += "/"
                labelsPos.add(label)
                ac += [label]
            labelDict[row[0]] = ac  # the original written form of every word
            print(row[0])
            modelDict[row[0]] = KeyedVectors.load_word2vec_format(directory + "\\" + row[0] + "Emb.txt", binary=False)
            print("Normalizing vectors...")
            modelDict[row[0]].init_sims(replace=True)
    print("Saving new dictionaries...")
    with open("labelDict.pk", 'wb') as fi:
        pickle.dump(labelDict, fi)
    with open("modelDict.pk", 'wb') as fi:
        pickle.dump(modelDict, fi)
    if useUni:
        with open("uniMap.pk", 'wb') as fi:
            pickle.dump(uniMap, fi)
else:
    print("Loading previously saved dictionaries...")
    with open("labelDict.pk", 'rb') as fi:
        labelDict = pickle.load(fi)
    with open("modelDict.pk", 'rb') as fi:
        modelDict = pickle.load(fi)
    if useUni:
        with open("uniMap.pk", 'rb') as fi:
            uniMap = pickle.load(fi)


# print(modelDict["English"].most_similar(positive=["kill", "eat"], negative=["feel"], topn=5))

def displayScatter(model, langList, feedRate=0.8, showAll=None, lines=True, deviation=0.01, perplexity=75,
                   threeD=False, showNum=None):
    if showAll is None:
        showAll = [False] * (len(langList) - 1)
    plt.figure(figsize=(6, 6))
    if threeD:
        ax = plt.axes(projection='3d')
    wordVectors = []
    twodim = []
    labelList = [labelDict[lang] for lang in langList]
    N = len(langList)
    hsvTuples = [(x * 1.0 / N, 1, 1) for x in range(N)]
    rgbTuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvTuples))
    if threeD:
        xyzPos = [[], [], []]
    else:
        xyPos = [[], []]
    boundary = round(len(labelDict[langList[0]]) * feedRate)
    scatterPos = []
    if threeD:
        n_comp = 3
    else:
        n_comp = 2
    tsneModel = TSNE(perplexity=perplexity, n_components=n_comp, init='pca', n_iter=2500,
                     random_state=15, metric="cosine", learning_rate=20)
    for (i, labels) in enumerate(labelList):  # 每种语言拆出来一个labelList
        wordVectors.append(np.array([model[w] for w in labels]))
        # twodim.append(PCA().fit_transform(wordVectors[i])[:, :2])
        twodim.append(tsneModel.fit_transform(wordVectors[i]))  # slicing numpy array

        color = rgbTuples[i]

        if i > 0 and not showAll[i - 1]:
            selection = twodim[i][boundary:]  # only display tested words
            '''
            if showNum is not None:
                tmpSelection = np.zeros(shape=(showNum, 2))
                for j in range(showNum):
                    tmpSelection[i] = selection[random.randrange(len(selection))]
                selection = tmpSelection
                print(selection)
            '''
        else:
            selection = twodim[i][:]
        if threeD:
            current = ax.scatter(selection[:, 0], selection[:, 1], selection[:, 2], edgecolors='face', color=color)
        else:
            current = plt.scatter(selection[:, 0], selection[:, 1], edgecolors='face', color=color)
        scatterPos.append(current)

        if langList[i] == "Chinese":
            font = "SimSun"
        elif langList[i] == "Japanese":
            font = "Yu Mincho"
        elif langList[i] == "Korean":
            font = "Malgun Gothic"
        elif langList[i] == "Tibetan":
            font = "Microsoft Himalaya"
        elif langList[i] == "Thai":
            font = "Leelawadee UI"
        elif langList[i] == "Sanskrit":
            font = "Nirmala UI"
        else:
            font = "Times New Roman"
        fprop = fm.FontProperties(fname=font)
        plt.rcParams['font.family'] = font
        if lines:
            xPos = []
            yPos = []
            if threeD:
                zPos = []
                for word, (x, y, z) in zip(labels, twodim[i][:]):
                    if i > 0 and showAll[i - 1]:
                        ax.text(x + deviation, y + deviation, z + deviation, word)
                    else:
                        if (x, y, z) in selection:
                            ax.text(x + deviation, y + deviation, z + deviation, word)
                    xPos.append(x)
                    yPos.append(y)
                    zPos.append(z)

                xyzPos[0].append(xPos)
                xyzPos[1].append(yPos)
                xyzPos[2].append(zPos)
            else:
                for word, (x, y) in zip(labels, twodim[i][:]):
                    if i > 0 and showAll[i - 1]:
                        plt.text(x + deviation, y + deviation, word)
                    else:
                        if (x, y) in selection:
                            plt.text(x + deviation, y + deviation, word)
                    xPos.append(x)
                    yPos.append(y)

                xyPos[0].append(xPos)
                xyPos[1].append(yPos)
        else:
            if threeD:
                for word, (x, y, z) in zip(labels, selection):
                    ax.text(x + deviation, y + deviation, z + deviation, word)
            else:
                for word, (x, y) in zip(labels, selection):
                    plt.text(x + deviation, y + deviation, word)
        plt.legend(scatterPos,
                   langList,
                   scatterpoints=1,
                   loc='lower left',
                   ncol=3,
                   fontsize=8)

    if lines:
        if threeD:
            print(xyzPos)
            linePos = np.transpose(xyzPos, (2, 0, 1))
            for langi in range(1, len(langList)):
                for j in range(len(linePos)):
                    spokes = []
                    if not showAll[langi - 1] and j < boundary:
                        continue
                    else:
                        spokes.append((linePos[j][0][0], linePos[j][0][langi]))
                        spokes.append((linePos[j][1][0], linePos[j][1][langi]))
                        spokes.append((linePos[j][2][0], linePos[j][2][langi]))
                        spokes.append('k-')
                    print(spokes)
                    ax.plot(*spokes, alpha=0.2)

        else:
            linePos = np.transpose(xyPos, (2, 0, 1))
            spokes = []
            for langi in range(1, len(langList)):
                for j in range(len(linePos)):
                    if not showAll[langi - 1] and j < boundary:
                        continue
                    else:
                        spokes.append((linePos[j][0][0], linePos[j][0][langi]))
                        spokes.append((linePos[j][1][0], linePos[j][1][langi]))
                        spokes.append('k-')
            plt.plot(*spokes, alpha=0.2)


def convertTuple(tup):
    s = ''
    for item in tup:
        s = s + " " + str(item)
    return s.strip()


def testLang(langs, englishComment=None, showWeight=False, feedRate=0.8,
             showGraph=False, deviation=0.01, showAll=None, showNum=None, lines=True, perplexity=10, printWords=True,
             censorRate=1.0, threeD=False, startRate=0, topNum=100, indexer=None, printDetails=True):
    global bilingual_model
    global source_model
    global target_model
    global useUni
    global uniMap
    if englishComment is None:
        englishComment = [True] * len(langs)
    models = [modelDict[lang] for lang in langs]
    sourceLang = langs[1:]
    targetLang = langs[0]
    if useUni:
        size = round(len(labelDict["Uni"]) * censorRate)
    else:
        size = round(len(labelDict[targetLang]) * censorRate)
    testNum = size - round(size * feedRate)
    if testNum == 0:
        return None
    if not printWords:
        report = []
    for (srci, src) in enumerate(sourceLang):  # every source language
        infer = []  # complete result of a single language
        success = 0
        with alive_bar(testNum, force_tty=True) as bar:
            for i in range(round(size * startRate),
                           round(size * startRate) + testNum):  # every word in test portion
                trans = []  # complete result of a single word
                caption = ""
                if useNeuralNetwork:
                    outVector = np.array(bilingual_model(torch.Tensor(models[1].get_vector(labelDict[src][i]))).tolist())
                    options = models[0].similar_by_vector(outVector, topn=topNum)
                else:
                    if useVecMap:
                        options = target_model.similar_by_vector(models[1].get_vector(labelDict[src][i]), topn=topNum)
                    else:
                        options = bilingual_model.most_similar([labelDict[src][i]], topn=topNum, indexer=indexer)

                for option in options:  # every option of this word
                    compare = option[0]  # possible word
                    weight = option[1]  # its weight
                    ind = labelDict[targetLang].index(compare)
                    if printWords:
                        if showWeight:
                            weight = ' %.2f' % weight
                        else:
                            weight = ""
                        if englishComment[0] and targetLang != "English":
                            enTrans = labelDict["English"][ind]
                            trans += [compare + " (" + enTrans + ")" + weight]
                        else:
                            trans += [compare + weight]
                    if useUni and compare == uniMap[labelDict[src][i]] or not useUni and ind == i:
                        caption = "*"
                if caption == "*":
                    if useUni:
                        print(labelDict[src][i], uniMap[labelDict[src][i]], options)
                    success += 1
                if printWords:
                    trans = ", ".join(trans)
                    srcWord = labelDict[src][i]
                    if englishComment[srci + 1] and src != "English":
                        infer += [caption + srcWord + " (" + labelDict["English"][i] + ") -> " + trans]
                    else:
                        infer += [caption + srcWord + " -> " + trans]
                bar()

        if printWords:
            if printDetails:
                print(''.join([sentence + '.\n' for sentence in infer]))
            print(src + " - Success Rate: " + str(success) + "/" + str(testNum) + " = " +
                  "{0:.0%}\n".format(success / testNum))
        else:
            # report.append("{0:.0%}".format(success / (testNum)))
            report.append(success / testNum)
            print("Model success", success, "test num", testNum)
    if showGraph and not useNeuralNetwork and not useVecMap:
        displayScatter(bilingual_model, langs,
                       deviation=deviation, feedRate=feedRate, showAll=showAll, lines=lines, perplexity=perplexity,
                       threeD=threeD, showNum=showNum)
    if not printWords:
        return report  # success rate of every source language


def compareLang(langs, englishComment=None, showWeight=False, feedRate=0.8,
                showGraph=False, deviation=0.01, showAll=None, showNum=None, lines=True, perplexity=10, printWords=True,
                censorRate=1.0, threeD=False, startRate=0, topNum=100, drawPlot=False, printDetails=True):
    global bilingual_model
    global indexer
    global source_model
    global target_model
    global uniMap
    if englishComment is None:
        englishComment = [True] * len(langs)
    models = [modelDict[lang] for lang in langs]

    sourceLang = langs[1:]
    targetLang = langs[0]
    if useUni:
        size = round(len(labelDict["Uni"]) * censorRate)
    else:
        size = round(len(labelDict[targetLang]) * censorRate)
    testNum = size - round(size * feedRate)
    trainLabels = []
    '''
    targetLabels = []
    sourceLabels = []
    for i in range(0, round(size * feedRate)):
        targetLabels.append(labelDict[langs[0]][i])
        sourceLabels.append(labelDict[langs[1]][i])
    random.shuffle(sourceLabels)
    
    for i in range(0, size):
        # add = tuple([labelDict[lang][i] for lang in langs])
        # add = (labelDict[langs[0]][i], labelDict[langs[1]][round(size*feedRate) - i - 1])
        add = (targetLabels[i], sourceLabels[i])
        if i < round(size * feedRate):
            trainLabels.append(add)
    '''

    print("Adding training labels...")
    for i in range(0, size):
        if round(size * startRate) <= i < (round(size * startRate) + testNum):
            continue
        if useNeuralNetwork:
            add = tuple([models[j].get_vector(labelDict[lang][i]) for j, lang in enumerate(langs)])
        else:
            if not useUni:
                add = []
                for lang in langs:
                    add.append(labelDict[lang][i])
                add = tuple(add)
            else:
                add = (uniMap[labelDict['Uni'][i]], labelDict['Uni'][i])

        trainLabels.append(add)
    # print(trainLabels)
    if drawPlot:
        if useNeuralNetwork:
            entries = np.array([np.concatenate((entry[0], entry[1]), axis=0).tolist() for entry in trainLabels])
        else:
            entries = np.array([np.concatenate((modelDict[targetLang].get_vector(entry[0]),
                                                modelDict[sourceLang[0]].get_vector(entry[1])), axis=0)
                                for entry in trainLabels])
        df = pd.DataFrame(entries)
        sns.color_palette("rocket")
        cmap = sns.diverging_palette(0, 255, as_cmap=True)
        sns.heatmap(df.corr(), cmap=cmap)
        plt.show()

    print("Training...")
    if useNeuralNetwork:
        bilingual_model = neuralNetwork.runTraining(trainLabels)
        indexer = None
    else:
        if useVecMap:
            with open("trainLabels.txt", 'w', encoding='utf-8') as f:
                for tup in trainLabels:
                    f.write(tup[1] + " " + tup[0] + "\n")
            command = "python vecmap-master\map_embeddings.py --supervised trainLabels.txt \"" \
                      + directory + "\\" + sourceLang[0] + "Emb.txt\" \"" \
                      + directory + "\\" + targetLang + "Emb.txt\" \"" \
                      + directory + "\\" + sourceLang[0] + "EmbM.txt\" \"" \
                      + directory + "\\" + targetLang + "EmbM.txt\" --cuda"
            subprocess.check_output(command, encoding='UTF-8', cwd=os.getcwd())
            source_model = KeyedVectors.load_word2vec_format(directory + "\\" + sourceLang[0] + "EmbM.txt",
                                                             binary=False)
            target_model = KeyedVectors.load_word2vec_format(directory + "\\" + targetLang + "EmbM.txt",
                                                             binary=False)
            indexer = None
            '''
            command = "python vecmap-master\eval_translation.py \""\
                      + directory + "\\" + sourceLang[0] + "EmbM.txt\" \"" \
                        + directory + "\\" + targetLang + "EmbM.txt\" -d trainLabels.txt"
            print(command)
            subprocess.check_output(command, encoding='UTF-8', cwd=os.getcwd())
            '''
            # with open("trainLabels.txt", 'w', encoding='utf-8') as f:
        else:
            bilingual_model = TranslationWordVectorizer(*models).fit(trainLabels)
            indexer = AnnoyIndexer(modelDict[targetLang], 1000)

            if drawPlot:
                ax = [None, None, None]
                fig, (ax[0], ax[1], ax[2]) = plt.subplots(ncols=3, sharey=True, figsize=(20, 8))
                for i in range(3):
                    vecs = np.array([bilingual_model.get_vector(labelDict[lang][i]) for lang in langs])
                    print(vecs)
                    sns.heatmap(vecs, ax=ax[i])
                plt.tight_layout()
                plt.show()

    report = testLang(langs, englishComment=englishComment, showWeight=showWeight, feedRate=feedRate,
                      showGraph=showGraph, deviation=deviation, showAll=showAll, showNum=showNum, lines=lines,
                      perplexity=perplexity, printWords=printWords, censorRate=censorRate, threeD=threeD,
                      startRate=startRate, topNum=topNum, indexer=indexer, printDetails=printDetails)
    if not printWords:
        return report  # success rate of every source language

    # print(bilingual_model.most_similar(["take_"], indexer=indexer))
    # print(bilingual_model.similar_by_vector(bilingual_model.get_vector("bite") - bilingual_model.get_vector("beißen")
    #                                        + bilingual_model.get_vector("verlassen")))

    '''
    wordLst = ["火柴"]
    # wordLst = ["拉", "辣", "辣椒", "蜡", "垃圾", "蜡烛", "喇叭"]
    # wordLst = ["balcony", "complete", "connect", "command", "coast"]
    # wordLst = ["雨", "雪", "水", "云朵", "冰"]
    # wordLst = ["热", "温", "火"]
    vectorLst = [bilingual_model.get_vector(w) for w in wordLst]
    vectorMean = np.mean(vectorLst, axis=0)
    print("Average word: ", bilingual_model.target.most_similar(vectorMean, topn=10, indexer=indexer))
    '''
    # print(bilingual_model.similarity("jouer", "play"))


def randomBaseline(feedRate, startRate, topNum, size, guessSize, targetLen):
    global useUni

    testNum = size - round(size * feedRate)
    success = 0

    if useUni:
        answers = list(range(targetLen)) * len(processAll.langAbb)
        random.shuffle(answers)

    for i in range(round(size * startRate),
                   round(size * startRate) + testNum):
        guessed = []
        for j in range(topNum):
            while True:
                guess = random.randrange(guessSize)
                if guess not in guessed:
                    break
            if useUni and guess == answers[i] or not useUni and guess == i:
                success += 1
                break
            else:
                guessed.append(guess)
    print("Baseline success", success, "test num", testNum)
    return success / testNum


def crossValidation(langs, totalFolds=10, testFolds=1, topNum=100, censorRate=1.0):
    global useUni
    targetLang = langs[0]
    targetLen = len(labelDict[targetLang])
    if useUni:
        size = round(len(labelDict["Uni"]) * censorRate)
    else:
        size = round(targetLen * censorRate)

    guessSize = targetLen  # the possible range of guesses does not change when we use uni
    report = []
    baselineReport = []
    difs = []
    n = totalFolds / testFolds
    for i in range(0, totalFolds, testFolds):
        if useUni:
            print("Test", str(int((i / testFolds) + 1)) + ":",
                  "from", "{0:.0%}".format(i / totalFolds),
                  "(" + labelDict["Uni"][round(size * i / totalFolds)] + ")",
                  "to", "{0:.0%}".format((i + testFolds) / totalFolds),
                  "(" + labelDict["Uni"][round(size * (i + testFolds) / totalFolds) - 1] + ")")
        else:
            print("Test", str(int((i / testFolds) + 1)) + ":",
                  "from", "{0:.0%}".format(i / totalFolds), "(" + labelDict[targetLang][round(size * i / totalFolds)] + ")",
                  "to", "{0:.0%}".format((i + testFolds) / totalFolds),
                  "(" + labelDict[targetLang][round(size * (i + testFolds) / totalFolds) - 1] + ")")
        feedRate = (totalFolds - testFolds) / totalFolds
        startRate = i / totalFolds
        report.append(compareLang(langs, feedRate=feedRate, startRate=startRate,
                                  printWords=False, topNum=topNum, censorRate=censorRate))
        baselineReport.append(randomBaseline(feedRate=feedRate,
                                             startRate=startRate, topNum=topNum, size=size, guessSize=guessSize, targetLen=targetLen))
        difs.append(report[int(i / testFolds)][0] - baselineReport[int(i / testFolds)])
        print("Model:", "{0:.2%}".format(report[int(i / testFolds)][0]),
              "Baseline:", "{0:.2%}".format(baselineReport[int(i / testFolds)]))
        # generateTable(targetLang, mode=4 + i / testFolds, topNum=100)
    mean = sum(difs) / n
    sd = math.sqrt(sum([pow(dif - mean, 2) for dif in difs]) / (n - 1))
    t = math.sqrt(n) * mean / sd

    print("Mean:", mean, "Standard deviation:", sd, "T-value:", t)


def strDict(dict):
    lst = []
    for (key, value) in dict:
        lst.append(key + " " + "{0:.0%}".format(value))
    s = ", ".join(lst)
    return s


def getZScore(freqDicts, topRank):
    rowPercentDicts = []
    rowMeans = {}  # for each phoneme, the averages of its percentages across the words
    rowN = {}  # for each phoneme, the number of percentages counted in the average
    # print("freqDicts", freqDicts)
    for row in range(len(freqDicts)):
        rowSum = sum([freqDicts[row][key] for key in freqDicts[row]])
        rowPercentDict = {}
        for key in freqDicts[row]:
            rowPercentDict[key] = freqDicts[row][key] / rowSum
        rowPercentDicts.append(rowPercentDict)
        # print(freqDicts[row], rowPercentDicts[row])
        for key in freqDicts[row]:
            if key not in rowMeans:
                rowMeans[key] = rowPercentDicts[row][key]
                rowN[key] = 1
            else:
                rowMeans[key] += rowPercentDicts[row][key]
                rowN[key] += 1
            # print("phoneme", key, "percent", rowPercentDicts[row][key])
    for key in rowMeans:
        rowMeans[key] /= rowN[key]
    rowSD = {}
    for row in range(len(rowPercentDicts)):
        for key in rowPercentDicts[row]:
            if key not in rowSD:
                rowSD[key] = pow(rowPercentDicts[row][key] - rowMeans[key], 2)
            else:
                rowSD[key] += pow(rowPercentDicts[row][key] - rowMeans[key], 2)
    for key in rowSD:
        rowSD[key] /= rowN[key]
        rowSD[key] = math.sqrt(rowSD[key])
    rowZScoreDicts = rowPercentDicts
    for row in range(len(rowZScoreDicts)):
        for key in rowZScoreDicts[row]:
            rowZScoreDicts[row][key] -= rowMeans[key]
            if rowSD[key] != 0:
                rowZScoreDicts[row][key] /= rowSD[key]
        rowZScoreDicts[row] = sorted(rowZScoreDicts[row].items(), key=lambda x: x[1], reverse=True)
        if topRank != 0:
            rowZScoreDicts[row] = rowZScoreDicts[row][:topRank]
    # print("n", rowN, "means", rowMeans, "sd", rowSD)
    return rowZScoreDicts


def generateHexVectors():
    global bilingual_model
    global indexer
    global hexLabels
    global hexVectors
    hexVectors = []
    # print(hexLabels)
    for i, words in enumerate(hexLabels):
        similarVectors = [modelDict["Semantics"].get_vector(w + '_') for w in words]
        vectorMean = np.mean(similarVectors, axis=0)
        hexVectors.append(vectorMean)
        print(hexNames[i] + '\n' + str(modelDict["Semantics"].most_similar(vectorMean)))


def mostCommon(lst):
    return max(set(lst), key=lst.count)


def generateTable(targetLang, sourceLang="Chinese", mode=1, topNum=100, topRank=5, genHexVec=True, transWord=True):
    global bilingual_model
    global indexer
    global hexNames
    global hexVectors
    targetDict = {}

    if targetLang != 'Semantics':
        with open(directory + '\\' + targetLang + 'V.txt', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, line in enumerate(reader):
                word, phones = line[0].split("  ")
                # targetDict[word] = phones.replace(' ', '')
                targetDict[word] = phones
    with open(directory + '\\wordTable.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        vowels = []
        targetTable = []
        rowFreqDicts = []
        colFreqDicts = []
        colHexDicts = []
        if genHexVec:
            generateHexVectors()
        for i, row in enumerate(reader):  # each row is a list
            if i == 0:
                consonants = row
            else:
                if i - 1 >= len(rowFreqDicts):
                    rowFreqDicts.append({})
                vowels.append(row[0])
                targetTable.append([])
                for col, words in enumerate(row[1:]):
                    if col >= len(colHexDicts):
                        colHexDicts.append({})
                    if col >= len(colFreqDicts):
                        colFreqDicts.append({})
                    if words == '':
                        targetTable[i - 1].append('')
                        continue
                    # print(words)
                    weights = [1 / len(w.replace('/', '')) for w in words.split(',')]
                    # print(weights)
                    if useNeuralNetwork:
                        vectorLst = [modelDict[sourceLang].get_vector(w) for w in words.split(',')]
                    else:
                        vectorLst = [bilingual_model.get_vector(w) for w in words.split(',')]
                    vectorMean = np.average(vectorLst, axis=0, weights=weights)
                    if useNeuralNetwork:
                        similarWords = modelDict[targetLang].similar_by_vector(vectorMean, topn=topNum)
                    else:
                        similarWords = bilingual_model.target.most_similar(vectorMean, topn=topNum, indexer=indexer)
                    weights = [tup[1] for tup in similarWords]
                    similarWords = [tup[0] for tup in similarWords]
                    if mode == 0:
                        if transWord:
                            translatedWords = [labelDict[targetLang][labelDict[sourceLang].index(w)] for w in
                                               words.split(',')]
                        else:
                            translatedWords = words.split(',')
                        addHexes = [np.argmax([float(val)
                                               for val in KeyedVectors.cosine_similarities(
                                bilingual_model.get_vector(w),
                                hexVectors).tolist()]) for w in translatedWords]
                        for hexInd in addHexes:
                            if hexNames[hexInd] not in colHexDicts[col]:
                                colHexDicts[col][hexNames[hexInd]] = 1
                            else:
                                colHexDicts[col][hexNames[hexInd]] += 1

                        vectorMean = np.average([bilingual_model.get_vector(w) for w in similarWords], axis=0,
                                                weights=[w / sum(weights) for w in weights])
                        similarities = [float(val) for val in
                                        KeyedVectors.cosine_similarities(vectorMean, hexVectors).tolist()]
                        # print(similarities)
                        hexCaption = hexNames[np.argmax(similarities)] + ': '

                        similarVectors = [bilingual_model.get_vector(w) for w in similarWords]
                        gregariousDeg = KeyedVectors.cosine_similarities(np.mean(similarVectors, axis=0),
                                                                         np.array(similarVectors))
                        if transWord:
                            translatedWords = [w[:-1] for w in translatedWords]
                        targetTable[i - 1].append(hexCaption + ','.join(
                            [''.join(tup) for tup in list(zip(translatedWords,
                                                              [' (' + hexNames[n] + ')' for n in addHexes]))]))
                    elif mode == 1:  # directly output the similar words
                        targetTable[i - 1].append(','.join(similarWords))
                    elif mode == 2:
                        # search for the common substring, starting from comparing the 1st word and the 2nd word
                        shareRange = [targetDict[similarWords[0]]]
                        commonEtym = ''
                        for j, w in enumerate(similarWords):
                            if j > 0:
                                shareRange.append(targetDict[w])
                                st = STree.STree(shareRange)
                                lcs = st.lcs()
                                # print(shareRange, lcs)
                                if lcs != '':
                                    commonEtym = lcs
                                else:
                                    break
                        targetTable[i - 1].append(commonEtym)
                        #  print(st.lcs())
                    elif mode == 3:  # count the frequency of each phoneme across the average words of similar words
                        if useNeuralNetwork:
                            vectorLst = [modelDict[targetLang].get_vector(w) for w in similarWords]
                        else:
                            vectorLst = [bilingual_model.get_vector(w) for w in similarWords]
                        # magWeights = softmax(weights)  # using softmax to magnify weights
                        vectorMean = np.average(vectorLst, axis=0, weights=[w / sum(weights) for w in weights])
                        if useNeuralNetwork:
                            essenceWord = modelDict[targetLang].similar_by_vector(vectorMean, topn=1)[0][0]
                        else:
                            essenceWord = bilingual_model.target.most_similar(vectorMean, topn=1, indexer=indexer)[0][0]
                        #  print(similarWords[0], vectorSum)
                        essencePhone = targetDict[essenceWord]
                        targetTable[i - 1].append(essencePhone.replace(' ', ''))
                        colCur = col
                        similarWords = [essenceWord]
                    elif mode > 3:  # count the frequency of each phoneme across the similar words
                        targetTable[i - 1].append(','.join([targetDict[w].replace(' ', '') for w in similarWords[:10]]))
                    if mode > 2:
                        for word in similarWords:
                            if mode > 3:

                                # merge some columns into one
                                if col == 3:
                                    colCur = 2
                                elif col == 8:
                                    colCur = 7
                                elif col == 11:
                                    colCur = 10
                                elif col == 13:
                                    colCur = 12
                                elif col == 15:
                                    colCur = 14
                                elif col == 21:
                                    colCur = 20
                                else:
                                    colCur = col

                                # colCur = col
                            for ph in targetDict[word].split(' '):
                                if ph not in rowFreqDicts[i - 1]:
                                    rowFreqDicts[i - 1][ph] = 1
                                else:
                                    rowFreqDicts[i - 1][ph] += 1
                                if ph not in colFreqDicts[colCur]:
                                    colFreqDicts[colCur][ph] = 1
                                else:
                                    colFreqDicts[colCur][ph] += 1
        if mode > 2:
            rowZScoreDicts = getZScore(freqDicts=rowFreqDicts, topRank=topRank)
            for i in range(len(targetTable)):
                targetTable[i].append(strDict(rowZScoreDicts[i]))
            colZScoreDicts = getZScore(freqDicts=colFreqDicts, topRank=topRank)
            targetTable.append([strDict(colZScoreDicts[i]) for i in range(len(colZScoreDicts))])
        if colHexDicts[0] != {}:
            colHexZScoreDicts = getZScore(freqDicts=colHexDicts, topRank=topRank)
            evaluations = [strDict(colHexZScoreDicts[i]) for i in
                           range(len(colHexZScoreDicts))]  # a row of hex names summing up each column of words
            print(colHexZScoreDicts, evaluations)
            targetTable.append(evaluations)

    with open('etymTable' + str(mode) + '.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(consonants)
        for i, row in enumerate(targetTable):
            if i < len(targetTable) - 1:
                writer.writerow([vowels[i]] + row)
            else:
                writer.writerow([""] + row)

    print("Done with output!")


def generateHex(langs, doubleHex=False, translate=False):
    global bilingual_model
    global hexLabels
    global hexNames
    global hexVectors

    generateHexVectors()
    if doubleHex:
        hexDoubleVectors = []
        hexDoubleNames = []
        for i, name1 in enumerate(hexNames):
            tmpLst = []
            for j, name2 in enumerate(hexNames):
                if name2 + ' ' + name1 in hexDoubleNames:
                    continue
                tmpLst.append(np.mean([hexVectors[i], hexVectors[j]], axis=0))
                hexDoubleNames.append(name1 + ' ' + name2)
            hexDoubleVectors += tmpLst
        hexNames = hexDoubleNames
        print(hexNames)
        hexVectors = hexDoubleVectors

    for lang in langs:
        print("Generating hexagrams for " + lang + "...")
        hexDict = [[] for i in range(len(hexNames))]
        for i, label in enumerate(labelDict[lang]):
            similarities = []
            for j in range(len(hexNames)):
                similarities.append(
                    float(KeyedVectors.cosine_similarities(hexVectors[j], [bilingual_model.get_vector(label)])))
            # print(similarities)
            if lang == "Semantics":
                if translate:
                    hexDict[np.argmax(similarities)].append(labelDict["Chinese"][i])
                else:
                    hexDict[np.argmax(similarities)].append(label[:-1])
            else:
                hexDict[np.argmax(similarities)].append(label)
        print([len(lst) for lst in hexDict])

        targetDict = {}
        if lang == 'Semantics':
            if translate:
                filename = directory + '\\ChineseV.txt'
            else:
                filename = directory + '\\EnglishV.txt'
        else:
            filename = directory + '\\' + lang + 'V.txt'
        with open(filename, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, line in enumerate(reader):
                word, phones = line[0].split("  ")
                # targetDict[word] = phones.replace(' ', '')
                targetDict[word] = phones

        freqDicts = [{} for i in range(len(hexNames))]
        phoneInv = {}
        phoneTotal = 0
        for i, labels in enumerate(hexDict):
            for label in labels:
                for ph in targetDict[label].split(' '):
                    if i == 0:
                        if ph not in phoneInv:
                            phoneInv[ph] = 1
                        else:
                            phoneInv[ph] += 1
                        phoneTotal += 1
                    if ph not in freqDicts[i]:
                        freqDicts[i][ph] = 1
                    else:
                        freqDicts[i][ph] += 1
        for ph in phoneInv:
            phoneInv[ph] /= phoneTotal
        phoneInv = sorted(phoneInv.items(), key=lambda x: x[1], reverse=True)
        print(phoneInv)
        ZScoreDicts = getZScore(freqDicts, topRank=10)
        # print([sum(dict.values()) for dict in freqDicts], freqDicts, ZScoreDicts)
        translateLabel = 'Translated' * (translate and lang == 'Semantics')
        with open(lang + 'Double' * doubleHex + 'Hex' + translateLabel + '.txt', 'w', encoding='utf-8') as f:
            for i in range(len(hexDict)):
                f.write(hexNames[i] + '\n' + strDict(ZScoreDicts[i]) + '\n' + ' '.join(hexDict[i]) + '\n')
        with open(lang + 'Double' * doubleHex + 'ZScore' + translateLabel + '.txt', 'w', encoding='utf-8') as f:
            for i in range(len(hexDict)):
                f.write(hexNames[i] + ': ' + strDict(ZScoreDicts[i]) + '\n')
            f.write('\n')


def experimentLang(langs, showGraph, printWords=True):
    maxJointScore = 0
    maxIndivScore = 0
    if printWords:
        print('Joint model')
        compareLang(langs,
                    showWeight=True, showGraph=showGraph, perplexity=90)
        print('Individual model')
        for i in range(1, len(langs)):
            compareLang([langs[0], langs[i]],
                        showWeight=True, showGraph=showGraph)
    else:
        jointReport = compareLang(langs,
                                  showWeight=True, showGraph=showGraph, perplexity=90, printWords=False)
        indivReport = []
        for i in range(1, len(langs)):
            indivReport.append(compareLang([langs[0], langs[i]],
                                           showWeight=True, showGraph=showGraph, printWords=False)[0])
            if jointReport[i - 1] > maxJointScore:
                maxJointScore = jointReport[i - 1]
            if indivReport[i - 1] > maxIndivScore:
                maxIndivScore = indivReport[i - 1]
            print(langs[i] + " " + "{0:.0%}".format(jointReport[i - 1]) + " " + "{0:.0%}".format(indivReport[i - 1])
                  + " " + str(jointReport[i - 1] < indivReport[i - 1]))
        print(" ")
    return maxJointScore, maxIndivScore


def iterateExperiment(langs):
    N = len(langs)
    combs = combinations(range(N), 2)
    maxJointScore = 0
    maxIndivScore = 0
    for comb in combs:
        allLang = list(range(N))
        langlst = []
        for i in range(len(allLang)):
            if i not in comb:
                langlst.append(i)
        for tgti in langlst:
            combNames = [langs[i] for i in comb]
            tgtName = langs[tgti]
            print(' '.join(combNames) + " > " + tgtName)
            scores = experimentLang([tgtName] + combNames, False, False)
            if scores[0] > maxJointScore:
                maxJointScore = scores[0]
            if scores[1] > maxIndivScore:
                maxIndivScore = scores[1]
    print("Max Joint Score: " + "{0:.0%}".format(maxJointScore) + "\nMax Individual Score: " +
          "{0:.0%}".format(maxIndivScore))


def langNetwork(langs, width=20, feedRate=0.8):
    N = len(langs)
    perms = permutations(range(N), 2)
    net = Network(height='650px', width='1500px')
    hsvTuples = [(x * 1.0 / N, 1, 1) for x in range(N)]
    rgbTuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsvTuples))
    rgbHex = ["#" + "".join("%02X" % round(i * 255) for i in rgb) for rgb in rgbTuples]
    net.add_nodes(list(range(N)), label=langs, color=rgbHex)
    nodes = []
    statSign = 2.464 / round(len(labelDict[langs[0]]) * (1 - feedRate))
    for perm in perms:
        indivReport = compareLang([langs[perm[0]], langs[perm[1]]], printWords=False, feedRate=feedRate)[0]
        rev = [perm[1], perm[0]]
        if indivReport > statSign:
            if rev in nodes and nodes[nodes.index(rev) + 2]:
                nodes[nodes.index(rev) + 1] += indivReport * width
                nodes[nodes.index(rev) + 2] = False
            else:
                nodes += [[perm[0], perm[1]], indivReport * width, True]
    nodesRes = []
    for i in range(0, len(nodes), 3):
        triple = tuple(nodes[i] + [nodes[i + 1]])
        nodesRes.append(triple)
    net.repulsion(node_distance=150, spring_length=200)
    net.add_edges(nodesRes)
    net.show('nodes.html')


def generateAverages(langs):
    labelLst = np.array([])
    vectorLst = []
    sns.set()
    takeNum = 5
    ax = [None] * takeNum
    fig, maps = plt.subplots(ncols=takeNum, sharey=True, figsize=(20, 8))
    for i, map in enumerate(maps):
        ax[i] = map
    for i, label in enumerate(labelDict["English"]):
        labelLst = np.append(labelLst, [label + "*"])
        vecs = np.array([modelDict[lang].get_vector(labelDict[lang][i]) for lang in langs])
        if i < takeNum:
            sns.heatmap(vecs, ax=ax[i])
        mean = np.mean(vecs, axis=0)
        vectorLst.append(mean)
    plt.tight_layout()
    plt.show()
    vectorLst = np.array(vectorLst)
    sns.heatmap(vectorLst)
    plt.show()
    print(labelLst, vectorLst)
    model = KeyedVectors(vectorLst.shape[1])
    model.add_vectors(labelLst, vectorLst)
    modelDict["Average"] = model
    labelDict["Average"] = labelLst.tolist()
    with open('average.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, label in enumerate(labelLst):
            writer.writerow([label, ','.join([str(val) for val in np.sign(vectorLst[i].tolist())])])


def main():
    # iterateExperiment(["Greek", "Russian", "Sanskrit", "German", "Spanish", "Italian", "French", "Irish", "English"])
    # iterateExperiment(["Chinese", "Vietnamese", "Korean", "Thai", "Indonesian"])
    # iterateExperiment(["Chinese", "Vietnamese", "Japanese", "Korean", "Tibetan", "Thai", "Indonesian"])

    # langNetwork(["Chinese", "Vietnamese", "Japanese", "Korean", "Tibetan", "Thai", "Indonesian"])
    '''
    langNetwork(["Greek", "Russian", "Hindi", "German", "Spanish", "Italian", "French", "Irish", "English"])



    langNetwork(['English', 'Hungarian', 'Finnish', 'Greek', 'Russian', 'German', 'Spanish', 'Italian', 'French', 'Irish', 'Welsh', 'Chinese', 'Vietnamese', 'Korean', 'Thai',
                'Indonesian', 'Turkish', 'Swahili', 'Hindi']
                , width=10)

    '''
    # experimentLang(["English", "Irish"], True)
    # experimentLang(["Korean", "Tibetan"], False)
    # experimentLang(["Japanese", "Korean", "Tibetan"], True)
    # experimentLang(["English", "Greek", "Russian", "Hindi", "German", "Spanish", "Italian", "French", "Irish"], False)
    # experimentLang(["Chinese", "Vietnamese", "Japanese", "Korean", "Tibetan", "Thai", "Indonesian"], True)
    # the first language is the target language, and the rest are the source languages.
    # Source languages can be shown partially (only the test portion).

    '''
    compareLang(["Turkish", "Japanese"],
                showWeight=True, englishComment=[True, True], showGraph=True, showAll=[False])
    compareLang(["Turkish", "Korean"],
                showWeight=True, englishComment=[True, True], showGraph=True, showAll=[False])
    compareLang(["Turkish", "Japanese", "Korean"],
                showWeight=True, englishComment=[True, True], showGraph=True, showAll=[False, False])
    
    compareLang(["English", "Russian", "Sanskrit", "German"],
                showWeight=True, englishComment=[True, False], showGraph=True, showAll=[False, False, False])
    
    '''
    # generateAverages(['English', 'Hungarian', 'Finnish', 'Greek', 'Russian', 'German', 'Spanish', 'Italian', 'French', 'Irish', 'Welsh', 'Chinese', 'Vietnamese', 'Korean', 'Thai',
    #            'Indonesian', 'Turkish', 'Swahili', 'Hindi'])
    # checkLst = ['poop', 'plant', 'smile', 'give', 'man', 'eat']
    # print([modelDict["Semantics"].most_similar(w + '_') for w in checkLst])
    # print(modelDict['Semantics'].similar_by_vector(modelDict['Semantics'].get_vector("please_") + modelDict['Semantics'].get_vector("bad_") - modelDict['Semantics'].get_vector("good_")))
    # usedLangs = ['Semantics', 'English', 'Hungarian', 'Finnish', 'Greek', 'Russian', 'German', 'Spanish', 'Italian', 'French', 'Irish', 'Welsh', 'Chinese', 'Vietnamese', 'Korean', 'Thai', 'Indonesian', 'Turkish', 'Swahili', 'Hindi']
    # usedLangs = ['Semantics', 'English', 'Chinese', 'French', 'German']
    # usedLangs = ['Semantics', 'Hungarian', 'Finnish', 'Greek', 'Russian', 'German', 'Spanish', 'Italian', 'French', 'Irish', 'Welsh', 'English', 'Chinese', 'Vietnamese', 'Korean', 'Thai', 'Indonesian', 'Turkish', 'Swahili', 'Hindi']

    usedLangs = ['Semantics', 'Vietnamese']
    if not useUni and needShuffle:
        shuffleLang(usedLangs)

    compareLang(usedLangs,
                showWeight=True, showGraph=False, showNum=100,
                perplexity=20, showAll=[False], threeD=False, printWords=True, printDetails=False,
                feedRate=0.99, censorRate=0.1, drawPlot=False, topNum=500, englishComment=[False, False])

    generateHex(usedLangs, doubleHex=True)
    '''
    
   
    testLang(["Chinese", "Turkish"],
             showWeight=True, showGraph=False, showNum=50,
             perplexity=20, showAll=[False], threeD=False, printWords=True,
             feedRate=0.9, censorRate=1)
    '''
    # generateTable("Semantics", mode=0, topNum=10, genHexVec=False, topRank=0, transWord=False)

    # generateTable("Irish", mode=8)
    # generateTable("English", mode=7, topNum=50)
    # generateTable("English", mode=4)
    # crossValidation(usedLangs, censorRate=0.005, topNum=100, totalFolds=10)
    # displayScatter(modelDict["Iti"], ["Iti"], deviation=0)
    # displayScatter(modelDict["Ifi"], ["Ifi"], deviation=0)
    # displayScatter(KeyedVectors.load_word2vec_format("ukrdict-3.7b-embeddings", binary=False, encoding='latin-1'),
    # ["English"])
    # displayScatter(modelDict["French"], ["French", "English"])


if __name__ == "__main__":
    main()

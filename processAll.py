import csv
import subprocess

import numpy as np


def purify(str):  # 规范化其他语言ipa用法
    str = " ".join([lt for lt in str.replace(" ", "")])
    str = str.replace(" ̪", "̪")
    str = str.replace(" ʰ", "ʰ")
    str = str.replace(" ʲ", "ʲ")
    str = str.replace(" ͡ ", "͡")
    str = str.replace(" ̃", "̃")
    str = str.replace(" ᵝ", "ᵝ")
    return str


directory = "D:\\Project\\Work in Progress\\phonetic-similarity-vectors-master"


def genDicts(lang, lst):
    labelsPos = set()
    with open(directory + "\\" + lang + 'V.txt', 'w', encoding='utf-8') as f:
        print(lst)
        for wi in range(len(lst)):
            w = lst[wi]
            res = purify(w)
            key = labelDict[lang][wi].rstrip().replace(",", "").replace(" ", "-")
            while key in labelsPos:
                key += "/"
            labelsPos.add(key)
            f.write(key + "  " + res + "\n")


langAbb = {'Hungarian': 'hu', 'Finnish': 'fi', 'Greek': 'el', 'Russian': 'ru', 'German': 'de', 'Spanish': 'es',
           'Italian': 'it', 'French': 'fr', 'Irish': 'ga', 'Welsh': 'cy', 'English': 'en', 'Vietnamese': 'vi',
           'Korean': 'ko',
           'Indonesian': 'id', 'Turkish': 'tr', 'Swahili': 'sw', 'Hindi': 'hi', 'Thai': 'th', 'Chinese': 'zh'}

global table
global labelDict


def genUniDict():
    global table
    zhDict = {}
    with open(directory + '\\ChineseV.txt', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            key, res = line.split('  ')
            zhDict[key] = res.rstrip()
    print(zhDict)
    open(directory + '\\UniV.txt', 'w', encoding='utf-8').close()  # clear file
    for row in table:
        labelsPos = set()
        lang = row[0]
        lst = row[1:]
        with open(directory + '\\UniV.txt', 'a', encoding='utf-8') as f:
            print(lst)
            for wi in range(len(lst)):
                w = lst[wi]
                res = purify(w)
                key = labelDict[lang][wi].rstrip().replace(",", "").replace(" ", "-")
                while key in labelsPos:
                    key += "/"
                labelsPos.add(key)
                f.write(key + "@" + langAbb[row[0]] + "  " + res + "\n")
    with open(directory + '\\UniV.txt', 'a', encoding='utf-8') as f:
        for key in zhDict:
            f.write(key + "@zh  " + zhDict[key] + "\n")


def genEmbeddings(lang):
    command = "python generate.py <" + lang + "V.txt >" + lang + "Emb.txt"
    subprocess.check_output(command, encoding='UTF-8', cwd=directory, shell=True)


def genTable(lang):
    command = "python generateTable.py <" + lang + "V.txt"
    subprocess.check_output(command, encoding='UTF-8', cwd=directory, shell=True)


def main():
    print(purify("apagat͡ɕt͡ɕʰat̪i"))
    global table
    global labelDict

    with open('Other.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        tableUnT = []
        for row in reader:  # each row is a list
            if '' in row:
                print(row)
                print("Encountered an empty string. Skip this line!")
                continue
            tableUnT.append(row)
        table = np.transpose(tableUnT)

    # arrange
    with open('nguasachV.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        labelsUnT = []
        for row in reader:  # each row is a list
            labelsUnT.append(row)
        labels = np.transpose(labelsUnT)

    labelDict = {}
    for row in labels:
        ac = []
        for w in row[1:]:
            ac += [w.rstrip().replace(" ", "-")]
        labelDict[row[0]] = ac  # the original written form of every word

    genUniNow = False

    if genUniNow:
        genUniDict()
    else:
        for i in range(len(table)):
            row = table[i]
            genDicts(row[0], row[1:])

    genEmbNow = False

    if genEmbNow:
        if genUniNow:
            print("Uni")
            genEmbeddings("Uni")
        else:
            print("Chinese")
            genEmbeddings("Chinese")
            # for row in table:
            #    print(row[0])
            #    genEmbeddings(row[0])
    else:
        print("Chinese")
        genTable("Chinese")
        '''
        for row in table:
            print(row[0])
            genTable(row[0])
        '''


if __name__ == "__main__":
    main()

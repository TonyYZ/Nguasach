# usage: python generate.py <cmudict-0.7b >cmudict-0.7b-embeddings
# reads a CMUDict-formatted text file on standard input, prints embeddings
# for each word on standard output.

# 请用python运行以下代码
# set PYTHONIOENCODING=utf8
# python generate.py <ukrdict-1.7b-with-vitz-nonce >ukrdict-3.7b-embeddings
# 其中1.7和3.7都只是版本号，可以随意修改

from collections import Counter
import sys

import numpy as np
from sklearn.decomposition import PCA

from featurephone import feature_bigrams
import csv

def normalize(vec):
    """Return unit vector for parameter vec.

    >>> normalize(np.array([3, 4]))
    array([ 0.6,  0.8])

    """
    if np.any(vec):
        norm = np.linalg.norm(vec)
        return vec / norm
    else:
        return vec

all_features = Counter()
entries = list()
consonantLst = []
vowelLst = []
vowels = ["i", "a", "e", "o", "u", "y", "ɑ", "ɯ", "ɤ", "ɔ", "ə"]
wordDict = {}
for i, line in enumerate(sys.stdin):
    if line.startswith(';'):
        continue
    line = line.strip()
    word, phones = line.split("  ")
    print("phones ", phones, file=sys.stderr)
    for syllable in phones.split("/"):
        parts = syllable.split()
        if parts[0] in vowels:
            consonant = "∅"
            vowel = "".join(parts)
        else:
            consonant = parts[0]
            if not parts[1:]:
                vowel = "∅"
            else:
                vowel = "".join(parts[1:])
        if consonant not in consonantLst:
            consonantLst.append(consonant)
            wordDict[consonant] = {}
        if vowel not in vowelLst:
            vowelLst.append(vowel)
        if vowel not in wordDict[consonant]:
            wordDict[consonant][vowel] = []
        if word not in wordDict[consonant][vowel]:
            wordDict[consonant][vowel].append(word + '')  # @zh is only for uni language
        #print(wordDict, file=sys.stderr)
print(consonantLst, vowelLst, wordDict, file=sys.stderr)
consonantLst.sort()
vowelLst.sort()
with open('wordTable.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    firstRow = [""]
    firstRow += consonantLst
    writer.writerow(firstRow)
    print(vowelLst, file=sys.stderr)
    for vowel in vowelLst:
        row = [vowel]
        for consonant in consonantLst:
            if vowel in wordDict[consonant]:
                row += [",".join(wordDict[consonant][vowel])]
            else:
                row += [""]
        writer.writerow(row)

print("Done with output!", file=sys.stderr)


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
for i, line in enumerate(sys.stdin):
    if line.startswith(';'):
        continue
    line = line.strip()
    word, phones = line.split("  ")
    print("phones ", phones, file=sys.stderr)
    features = Counter(feature_bigrams(phones.split()))
    entries.append((word, features))
    all_features.update(features.keys())

print("entries:", len(entries), file=sys.stderr)

filtfeatures = sorted([f for f, count in all_features.items() \
                       if count >= 2])
#print(filtfeatures, len(filtfeatures), len(filtfeatures[0]), file=sys.stderr)
print("feature count:", len(filtfeatures), file=sys.stderr)
print("performing PCA...", file=sys.stderr)

arr = np.array([normalize([i.get(j, 0) for j in filtfeatures]) \
                for word, i in entries])
n_components = 300
pca = PCA(n_components=n_components, whiten=True)
transformed = pca.fit_transform(arr)

del PCA
del arr

print("output...", file=sys.stderr)

# only for Nguasach project!!
print(str(len(entries)) + " " + str(n_components))

for i in range(len(entries)):
    word = entries[i][0]
    nums = " ".join(["%0.6f" % num for num in transformed[i]])
    if i < len(entries) - 1:
        print(" ".join([word, nums]))
    else:
        print(" ".join([word, nums]), end="")

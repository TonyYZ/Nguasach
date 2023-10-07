from sklearn.decomposition import PCA
import numpy as np

labels = []
arr = []

with open('model.txt', encoding='utf-8') as f:
    content = f.readlines()
    for line in content:
        elems = line.split()
        labels.append(elems[0])
        vector = [float(val) for val in elems[1:]]
        arr.append(vector)

print("performing PCA...")
arr = np.array(arr)
n_components = 50
pca = PCA(n_components=n_components, whiten=True)
transformed = pca.fit_transform(arr)

print("output...")
directory = "D:\\Project\\Work in Progress\\phonetic-similarity-vectors-master"
with open(directory + '\\' + 'SemanticsEmb.txt', 'w', encoding='utf-8') as f:
    f.write(str(len(arr)) + ' ' + str(n_components) + '\n')
    for i, entry in enumerate(transformed):
        nums = " ".join(["%0.6f" % val for val in entry])
        if i < len(arr) - 1:
            f.write(" ".join([labels[i], nums]) + '\n')
        else:
            f.write(" ".join([labels[i], nums]))
print(len(labels))
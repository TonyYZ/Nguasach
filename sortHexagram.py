import numpy as np

hexDict = {}

def generate_binary_combinations(n):
    for i in range(2**n):
        binary_str = bin(i)[2:].zfill(n)
        digits = list(map(int, binary_str))
        hexDict[repr(digits)] = []

generate_binary_combinations(3)
keyword = 'Irish'
directory = "D:\\Project\\Work in Progress\\phonetic-similarity-vectors-master"
with open(directory + '\\' + keyword + 'Emb.txt', encoding='utf-8') as f:
    content = f.readlines()
    content = content[1:]
    for line in content:
        words = line.split(' ')
        label = words[0]
        hexagram = [round((np.sign(float(num.rstrip())) + 1) / 2) for num in words[1:]]
        # print(hex, words[1:])
        hexDict[repr(hexagram)].append(label)

with open(keyword + 'Hex.txt', 'w', encoding='utf-8') as f:
    for hexName in hexDict.keys():
        f.write(hexName + '\n' + ' '.join(hexDict[hex]) + '\n')
print(hexDict)
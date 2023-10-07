# import these modules
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))

# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos="a"))

lemLst = []
total = 0
with open('things3.txt', newline='', encoding='utf-8') as f:
    content = f.readlines()
    for row in content:
        lemWord = lemmatizer.lemmatize(row.rstrip()[:-1])
        lemLst.append(lemWord)
        if row.rstrip()[:-1] != lemWord:
            total += 1
            print(row.rstrip()[:-1], lemWord)
print(total)

with open('lemmatized.txt', 'w', newline='', encoding='utf-8') as f:
    f.write('\n'.join(lemLst))
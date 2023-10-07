from frame_semantic_transformer import FrameSemanticTransformer
import re
from lemminflect import getAllLemmas, getAllInflections

def importDoubleHex():
    with open('SemanticsDoubleHex.txt', encoding='utf-8') as f:
        content = f.readlines()
        doubleHexDict = {}
        for i in range(0, len(content), 3):
            hexCombo = content[i].split(' ')
            hexCombo[1] = hexCombo[1].rstrip()
            for j, w in enumerate(content[i + 2].split(' ')):
                    w = w.rstrip()
                    doubleHexDict[w] = (hexCombo[0], hexCombo[1])
    return doubleHexDict


doubleHexDict = importDoubleHex()

frameTransformer = FrameSemanticTransformer()

passage = "If it cooks at 400 for an hour, it 'll be nothing but a pile of ash!"

wordDict = {}  # a dictionary where the key is the word index and the value is the word
for m in re.finditer(r'\S+', passage):
    index, item = m.start(), m.group()
    wordDict[index] = item.replace('.', '').replace(',', '')
print(wordDict)
result = frameTransformer.detect_frames(passage)

print(f"Results found in: {result.sentence}")

for frame in result.frames:
    print(f"FRAME: {frame.name}")
    keyword = wordDict[frame.trigger_location]
    if keyword not in doubleHexDict:
        options = [tup2[0] for tup2 in [getAllInflections(tup[0]) for tup in getAllLemmas(keyword).values()][0].values()]
        for keyword in options:
            if keyword in doubleHexDict:
                break
        if keyword not in doubleHexDict:
            print("Keyword:", keyword)
            keyword = ''
    if keyword != '':
        print("Keyword:", keyword, "Hex:", doubleHexDict[keyword])
    for element in frame.frame_elements:
        print(f"{element.name}: {element.text}")

import re
import unicodedata

with open('Thai.txt', encoding='utf-8') as f:
    text = f.readlines()

print(text)

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

reslst = ["Thai"]
for ln in text:
    found = re.findall(r"\t.*", ln)
    replaced = found[0].replace("\t","").replace("ʰ","!")
    res = remove_accents(replaced).replace("!", "ʰ").replace("ː", "").replace("tɕ", "t͡ɕ")
    reslst += [res]

with open('ThaiV.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(reslst))

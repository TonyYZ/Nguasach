import csv
import subprocess
import bophono
import numpy as np
from alive_progress import alive_bar
import main
import transcriber_data

'''
session = HTMLSession()
def irishIPA(text):
    url = "https://gphemsley.org/linguistics/ga/ortho2ipa/?text=" + text
    r = session.get(url)
    sel = "body > p.output"
    result = r.html.find(sel)
    return result[0].text
'''

def sanskritIPA(text):
    return transcriber_data.transcribe(text)


print(sanskritIPA("परन्तु"))

options = {
    'aspirateLowTones': True
}

converter = bophono.UnicodeToApi(schema="MST", options=options)


def tibetanIPA(text):
    return converter.get_api(text)


print(tibetanIPA("བོད་སྐད།།"))
dir = "D:\\Project\\Nguasach\\eSpeak NG\\"


japdict = {}
with open(r'D:\Project\Nguasach\ipa-dict-master\ipa-dict-master\data\ja.txt', encoding='utf-8') as f:
    for row in f:
        key = row.split()[0]
        value = row.split()[1].replace(',', '')[1:-1]
        japdict[key] = value

thaiV = []
with open(r'D:\Project\Nguasach\thaiV.txt', encoding='utf-8') as f:
    for row in f:
        thaiV.append(row.rstrip())
print(thaiV)

# arrange
with open('nguasachVOnlyE.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    arranged = []
    for row in reader:  # each row is a list
        arranged.append(row)
    arrangedT = np.transpose(arranged)


# create phonetics transcriptions
f = open("resultV.csv","w")
f.close()
print()

with alive_bar(len(arrangedT) * (len(arrangedT[0]) - 1), force_tty=True) as bar:
    for i in range(len(arrangedT)):
        #lang = "fr"
        lang = main.langLst2[i]

        if lang == "san" or lang == "bod" or lang == "ja" or lang == "th":
            with open('resultV.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                row = [arrangedT[i, 0]]  # start with lang name
                if lang == "th":
                    row += thaiV
                elif lang == "san":
                    for w in arrangedT[i, 1:]:
                        row += [sanskritIPA(w)]
                elif lang == "ja":
                    for w in arrangedT[i, 1:]:
                        row += [japdict[w.rstrip()]]
                else:
                    for w in arrangedT[i, 1:]:
                        out = tibetanIPA(w)
                        # print(out)
                        row += [out]
                writer.writerow(row)
        else:
            proc = [arrangedT[i, 0]]  # start with lang name
            for w in arrangedT[i, 1:]:
                with open(dir + 'readyV.txt', 'w', newline='', encoding='utf-8') as f:  # fill with dif languages successively
                    f.write(w)
                    label = w
                command = "espeak-ng -v " + lang + " -q --tie --ipa -f readyV.txt > processedV.txt"
                subprocess.check_output(command, encoding='UTF-8', cwd=dir, shell=True)
                with open(dir + 'processedV.txt', encoding='utf-8') as f:
                    w = f.readlines()[0].strip()
                    # print(label + ": " + w)
                    proc += [w]
                bar()
            with open('resultV.csv', 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(proc)
            print("Done with translations of " + arrangedT[i, 0] + "!")
        bar()

print("Done with output!")
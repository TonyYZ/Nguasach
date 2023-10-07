table = []
voiced = ['b', 'd', 'g', 'ɐ', 'e', 'u', 'i', 'o', 'ɯ', 'ɛ', 'ʌ', 'w', 'j', 'ɫ', 'm', 'n']
with open('KoreanRaw.txt', encoding='utf-8') as f:
    content = f.readlines()
    content = content[1:]
    #print(content)
    for line in content:
        line = line.rstrip()
        tmpLine = line
        result = 0
        while result != -1:
            result = line.rfind("q")
            if result != -1:
                if 0 < result < len(line) - 1 and line[result - 1] in voiced and line[result + 1] in voiced:
                    line = line[:result] + '9' + line[result + 1:]  # q to g
                else:
                    line = line[:result] + '8' + line[result + 1:]  # q to k
        result = 0
        while result != -1:
            result = line.rfind("ɡ")
            if result != -1:
                if 0 < result < len(line) - 1:
                    line = line[:result] + '9' + line[result + 1:]  # g to g
                else:
                    line = line[:result] + '8' + line[result + 1:]  # g to k
        result = 0
        while result != -1:
            result = line.rfind("d")
            if result != -1:
                if 0 < result < len(line) - 1:
                    line = line[:result] + '7' + line[result + 1:]  # d to d
                else:
                    line = line[:result] + '6' + line[result + 1:]  # d to t

        line = line.replace('8', 'k').replace('9', 'ɡ').replace('7', 'd').replace('6', 't')
        table.append(line)
        print(line)

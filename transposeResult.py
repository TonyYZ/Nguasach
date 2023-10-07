import csv
import numpy as np

with open('resultV.csv', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    tableUnT = []
    for row in reader:  # each row is a list
        tableUnT.append(row)
    table = np.transpose(tableUnT)
print(table)
with open('resultVT.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in table:
        writer.writerow([s.rstrip() for s in row])
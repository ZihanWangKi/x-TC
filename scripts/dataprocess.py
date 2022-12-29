import json
import os

with open("../data/NYT-fine/unlabeled.json", 'r') as f:
    train_list = json.load(f)

texts = []
labels = []
for i in range(len(train_list)):
    text, label = train_list[i]
    texts.append(text)
    labels.append(label)

print(labels[:10])

with open("../data/NYT-fine/labels.txt", 'w') as f:
    for i in range(len(labels)):
        f.write(str(labels[i]))
        f.write("\n")
with open("../data/NYT-fine/dataset.txt", 'w') as f:
    for i in range(len(texts)):
        f.write(str(texts[i]))
        f.write("\n")
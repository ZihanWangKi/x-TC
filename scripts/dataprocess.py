import json
import os

with open("../data/20NF/unlabeled.json", 'r') as f:
    train_list = json.load(f)

texts = []
labels = []
for i in range(len(train_list)):
    text, label = train_list[i]
    texts.append(text)
    labels.append(label)


os.system("mkdir ../data/20News-fine")
with open("../data/20News-fine/labels.txt", 'w') as f:
    for i in range(len(labels)):
        f.writelines(str(labels[i]))
with open("../data/20News-fine/dataset.txt", 'w') as f:
    for i in range(len(texts)):
        f.writelines(str(texts[i]))
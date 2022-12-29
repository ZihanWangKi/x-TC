import json
import os

with open("../data/20News-fine/unlabeled.json", 'r') as f:
    train_list = json.load(f)

texts = []
labels = []
for i in range(len(train_list)):
    text, label = train_list[i]
    if label == 6 or label == 18 or label == 19:
        continue
    elif label > 6:
        label -= 1
    text = text.replace('\n', '\\n')
    text = text.replace('\t', '\\t')
    text = text.replace('\r', '\\r')
    text = text.replace('\'', "\\'")
    text = text.replace('\"', '\\"')
    text = text.replace('\f', '\\f')
    texts.append(text)
    labels.append(label)

print(labels[:10])

with open("../data/20News-fine/labels.txt", 'w') as f:
    for i in range(len(labels)):
        f.write(str(labels[i]))
        f.write("\n")
with open("../data/20News-fine/dataset.txt", 'w') as f:
    for i in range(len(texts)):
        f.write(str(texts[i]))
        f.write("\n")
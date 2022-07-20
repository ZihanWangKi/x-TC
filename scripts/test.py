

with open("seed_words.txt", mode='r', encoding='utf-8') as text_file:
    seeds = list(map(lambda x: x.strip(), text_file.readlines()))
for i in range(len(seeds)):
    print(seeds[i].split(' '))
print(seeds)
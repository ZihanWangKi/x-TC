import os
import pickle
import json
import pandas as pd
from datasets import load_dataset, list_datasets

def split_data(args):
    if args.dataset in list_datasets():
        dataset = load_dataset(args.dataset, split="train")
        train_test = dataset.train_test_split(test_size=args.test_size, train_size=args.train_size,
                                 shuffle=True, seed=args.random_state)
        train_set = train_test["train"]
        test_set = train_test["test"]
        print(set)
    else:
        ...
    print("Finish data split!")
    print("train size: {}, test size: {}".format(len(train_set), len(test_set)))
    return train_set, test_set


def run_method(args, train_set, test_set):
    if args.method == "X-Class":
        assert args.class_names
        assert args.seed_words==False
        os.system("mkdir -p ../methods/X-Class/data/datasets/{}".format(args.dataset))
        os.system("cp class_names.txt ../methods/X-Class/data/datasets/{}/classes.txt".format(args.dataset))
        with open("../methods/X-Class/data/datasets/{}/dataset.txt".format(args.dataset), "w") as f:
            for line in train_set["text"]:
                f.write(line)
                f.write("\n")
        with open("../methods/X-Class/data/datasets/{}/labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.system("mkdir -p ../methods/X-Class/data/datasets/{}_test".format(args.dataset))
        os.system("cp class_names.txt ../methods/X-Class/data/datasets/{}_test/classes.txt".format(args.dataset))
        with open("../methods/X-Class/data/datasets/{}_test/dataset.txt".format(args.dataset), "w") as f:
            for line in test_set["text"]:
                f.write(line)
                f.write("\n")
        with open("../methods/X-Class/data/datasets/{}_test/labels.txt".format(args.dataset), "w") as f:
            for line in test_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.chdir("../methods/X-Class/scripts")
        os.system("chmod -R 777 run.sh")
        os.system("chmod -R 777 run_train_text_classifier.sh")
        os.system("./run.sh {} {}".format(args.gpu, args.dataset))
    elif args.method == "ConWea":
        assert args.class_names==False
        assert args.seed_words==True
        os.system("mkdir -p ../methods/ConWea/data/{}".format(args.dataset))
        os.system("mkdir -p ../methods/ConWea/tmp")
        train={
            "sentence": train_set["text"],
            "label": train_set[args.label_name]
        }
        df = pd.DataFrame(train)
        with open("../methods/ConWea/data/{}/df.pkl".format(args.dataset),'wb') as f:
            pickle.dump(df, f)
        with open("seed_words.txt", mode='r', encoding='utf-8') as f:
            seeds = list(map(lambda x: x.strip(), f.readlines()))
        seed_words={}
        for i in range(len(seeds)):
            seed_words[i] = seeds[i].split(' ')
        with open("../methods/ConWea/data/{}/seedwords.json".format(args.dataset), 'w') as f:
            json.dump(seed_words, f, indent=2)
        ...
        os.chdir("../methods/ConWea")
        os.system("python contextualize.py --dataset_path data/{}/ --temp_dir tmp/ --gpu_id {}"
                  .format(args.dataset, args.gpu))
        os.system("python train.py --dataset_path data/{}/ --gpu_id {}"
                  .format(args.dataset, args.gpu))
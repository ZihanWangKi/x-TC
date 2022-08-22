import os
import pickle
import json
import pandas as pd
from datasets import load_dataset, list_datasets
import csv

def split_data(args):
    if args.dataset in list_datasets():
        test_set = load_dataset(args.dataset, split="test")
        dataset = load_dataset(args.dataset, split="train")
        if args.train_size < 1.0:
            train_test = dataset.train_test_split(train_size=args.train_size,
                                                shuffle=False, seed=args.random_state)
            train_set = train_test["train"]
        else:
            train_set = load_dataset(args.dataset, split="train")

        #test_set = train_test["test"]
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
        label = [str(i) for i in train_set[args.label_name]]
        train={
            "sentence": train_set["text"],
            "label": label
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

        label = [str(i) for i in test_set[args.label_name]]
        test = {
            "sentence": test_set["text"],
            "label": label
        }
        test_df = pd.DataFrame(test)
        with open("../methods/ConWea/data/{}/test_df.pkl".format(args.dataset),'wb') as f:
            pickle.dump(test_df, f)

        os.chdir("../methods/ConWea")
        os.system("rm -rf tmp")
        os.system("mkdir tmp")
        os.system("python contextualize.py --dataset_path data/{}/ --temp_dir tmp/ --gpu_id {}"
                  .format(args.dataset, args.gpu))
        os.system("python train.py --dataset_path data/{}/ --gpu_id {}"
                  .format(args.dataset, args.gpu))
    elif args.method == "LOTClass":
        assert args.class_names == True
        assert args.seed_words == False
        os.system("rm -rf ../methods/LOTClass/datasets/{}".format(args.dataset))
        os.system("mkdir -p ../methods/LOTClass/datasets/{}".format(args.dataset))
        os.system("cp class_names.txt ../methods/LOTClass/datasets/{}/label_names.txt".format(args.dataset))
        with open("../methods/LOTClass/datasets/{}/train.txt".format(args.dataset), "w") as f:
            for line in train_set["text"]:
                f.write(line)
                f.write("\n")
        with open("../methods/LOTClass/datasets/{}/train_labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/LOTClass/datasets/{}/test.txt".format(args.dataset), "w") as f:
            for line in test_set["text"]:
                f.write(line)
                f.write("\n")
        with open("../methods/LOTClass/datasets/{}/test_labels.txt".format(args.dataset), "w") as f:
            for line in test_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.chdir("../methods/LOTClass")
        os.system("CUDA_VISIBLE_DEVICES={} python src/train.py --dataset_dir datasets/{}/ --test_file test.txt --test_label_file test_labels.txt --train_batch_size 32 --accum_steps 4 --gpus 1 --random_state {}"
                  .format(args.gpu, args.dataset, args.random_state))
    elif args.method == "WeSTClass":
        if args.class_names:
            assert args.seed_words == False
            os.system("rm -rf ../methods/WeSTClass/{}".format(args.dataset))
            os.system("mkdir -p ../methods/WeSTClass/{}".format(args.dataset))
            with open("class_names.txt", mode='r', encoding='utf-8') as f:
                names = list(map(lambda x: x.strip(), f.readlines()))
            with open("../methods/WeSTClass/{}/classes.txt".format(args.dataset), "w") as f:
                id = 0
                for line in names:
                    f.write(str(id) + ":" + line)
                    f.write("\n")
                    id += 1

            train = {
                "label": train_set[args.label_name],
                "sentence": train_set["text"]
            }
            df = pd.DataFrame(train)
            df.to_csv("../methods/WeSTClass/{}/dataset.csv".format(args.dataset), header=False, index=False)
            test = {
                "label": test_set[args.label_name],
                "sentence": test_set["text"]
            }
            df = pd.DataFrame(test)
            df.to_csv("../methods/WeSTClass/{}/dataset_test.csv".format(args.dataset), header=False, index=False)

            os.chdir("../methods/WeSTClass")
            os.system(
                "CUDA_VISIBLE_DEVICES={} python main.py --dataset {} --random_state {}"
                .format(args.gpu, args.dataset, args.random_state))
        elif args.seed_words == True:
            assert args.class_names == False
            os.system("rm -rf ../methods/WeSTClass/{}".format(args.dataset))
            os.system("mkdir -p ../methods/WeSTClass/{}".format(args.dataset))
            with open("seed_words.txt", mode='r', encoding='utf-8') as f:
                seeds = list(map(lambda x: x.strip(), f.readlines()))
            with open("../methods/WeSTClass/{}/keywords.txt".format(args.dataset), "w") as f:
                id = 0
                for line in seeds:
                    f.write(str(id) + ":" + line.replace(' ', ','))
                    f.write("\n")
                    id += 1

            train = {
                "label": train_set[args.label_name],
                "sentence": train_set["text"]
            }
            df = pd.DataFrame(train)
            df.to_csv("../methods/WeSTClass/{}/dataset.csv".format(args.dataset), header=False, index=False)
            test = {
                "label": test_set[args.label_name],
                "sentence": test_set["text"]
            }
            df = pd.DataFrame(test)
            df.to_csv("../methods/WeSTClass/{}/dataset_test.csv".format(args.dataset), header=False, index=False)

            os.chdir("../methods/WeSTClass")
            os.system(
                "CUDA_VISIBLE_DEVICES={} python main.py --dataset {} --sup_source keywords --random_state {}"
                .format(args.gpu, args.dataset, args.random_state))
        elif ...:
            ...
    elif args.method == "ClassKG":
        assert args.class_names == False
        assert args.seed_words == True
        ...
    elif args.method.startswith("gpt"):
        with open("../methods/GPT/data/{}/test.txt".format(args.dataset), "w") as f:
            for line in test_set["text"]:
                f.write(str(line))
                f.write("\n")
        os.chdir("../methods/GPT")
        os.system(
            "CUDA_VISIBLE_DEVICES={} python score.py {} --model {} --split test --seed {}"
            .format(args.gpu, args.dataset, args.method, args.random_state))
        #os.system(
        #    "CUDA_VISIBLE_DEVICES={} python main.py --dataset {} --sup_source keywords --random_state {}"
        #    .format(args.gpu, args.dataset, args.random_state))


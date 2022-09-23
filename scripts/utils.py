import os
import pickle
import json
import pandas as pd
import numpy as np
from datasets import load_dataset, list_datasets
import csv

def split_data(args):
    if args.dataset in list_datasets():
        test_set = load_dataset(args.dataset, split=args.split)
        dataset = load_dataset(args.dataset, split="train")
        if args.train_size < 1.0:
            train_test = dataset.train_test_split(train_size=args.train_size,
                                                shuffle=False, seed=args.random_state)
            train_set = train_test["train"]
        else:
            train_set = load_dataset(args.dataset, split="train")

        #test_set = train_test["test"]
    else:
        ...
    print("Finish data split!")
    print("Sampled instance:", test_set[0])
    print("train size: {}, test size: {}".format(len(train_set), len(test_set)))
    return train_set, test_set


def run_method(args, train_set, test_set):
    if args.method == "X-Class":
        assert args.class_names
        assert args.seed_words==False
        os.system("mkdir -p ../methods/X-Class/data/datasets/{}".format(args.dataset))
        os.system("cp class_names.txt ../methods/X-Class/data/datasets/{}/classes.txt".format(args.dataset))
        with open("../methods/X-Class/data/datasets/{}/dataset.txt".format(args.dataset), "w") as f:
            for line in train_set[args.text_name]:
                f.write(line)
                f.write("\n")
        with open("../methods/X-Class/data/datasets/{}/labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.system("mkdir -p ../methods/X-Class/data/datasets/{}_test".format(args.dataset))
        os.system("cp class_names.txt ../methods/X-Class/data/datasets/{}_test/classes.txt".format(args.dataset))
        with open("../methods/X-Class/data/datasets/{}_test/dataset.txt".format(args.dataset), "w") as f:
            for line in test_set[args.text_name]:
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
            "sentence": train_set[args.text_name],
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
            "sentence": test_set[args.text_name],
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
            for line in train_set[args.text_name]:
                f.write(line)
                f.write("\n")
        with open("../methods/LOTClass/datasets/{}/train_labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/LOTClass/datasets/{}/test.txt".format(args.dataset), "w") as f:
            for line in test_set[args.text_name]:
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
                "sentence": train_set[args.text_name]
            }
            df = pd.DataFrame(train)
            df.to_csv("../methods/WeSTClass/{}/dataset.csv".format(args.dataset), header=False, index=False)
            test = {
                "label": test_set[args.label_name],
                "sentence": test_set[args.text_name]
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
                "sentence": train_set[args.text_name]
            }
            df = pd.DataFrame(train)
            df.to_csv("../methods/WeSTClass/{}/dataset.csv".format(args.dataset), header=False, index=False)
            test = {
                "label": test_set[args.label_name],
                "sentence": test_set[args.text_name]
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
        os.system("mkdir -p ../methods/ClassKG/data/processed/{}".format(args.dataset))

        with open("seed_words.txt", mode='r', encoding='utf-8') as f:
            seeds = list(map(lambda x: x.strip(), f.readlines()))
        seed_words = {}
        for i in range(len(seeds)):
            seed_words[i] = seeds[i].split(' ')
        with open("../methods/ClassKG/data/processed/{}/keywords.json".format(args.dataset), 'w') as f:
            json.dump(seed_words, f, indent=2)

        label = [str(i) for i in test_set[args.label_name]]
        test = {
            "sentence": test_set[args.text_name],
            "label": label
        }

        train_list = []
        for i in range(len(train_set[args.text_name])):
            train_list.append([train_set[args.text_name][i], train_set[args.label_name][i]])
        test_list = []
        for i in range(len(test_set[args.text_name])):
            test_list.append([test_set[args.text_name][i], test_set[args.label_name][i]])
        with open("../methods/ClassKG/data/processed/{}/unlabeled.json".format(args.dataset), 'w') as f:
            json.dump(train_list, f, indent=2)
        with open("../methods/ClassKG/data/processed/{}/test.json".format(args.dataset), 'w') as f:
            json.dump(test_list, f, indent=2)

        os.chdir("../methods/ClassKG")
        os.system("python task/pipeline.py --dataset {} --random_state {}".format(args.dataset, args.random_state))
    elif args.method.startswith("gpt") and args.additional_method == None:
        assert args.prompt == True
        assert args.class_names == True
        assert args.seed_words == False
        if args.n_shot == 0:
            os.system("mkdir -p ../methods/GPT/data/{}".format(args.dataset))
            os.system("cp class_names.txt ../methods/GPT/data/{}/class_names.txt".format(args.dataset))
            os.system("cp prompt.txt ../methods/GPT/data/{}/prompt.txt".format(args.dataset))
            with open("../methods/GPT/data/{}/train.txt".format(args.dataset), "w") as f:
                for line in train_set[args.text_name]:
                    f.write(str(line))
                    f.write("\n")
            with open("../methods/GPT/data/{}/train_labels.txt".format(args.dataset), "w") as f:
                for line in train_set[args.label_name]:
                    f.write(str(line))
                    f.write("\n")
            with open("../methods/GPT/data/{}/test.txt".format(args.dataset), "w") as f:
                 for line in test_set[args.text_name]:
                    f.write(str(line))
                    f.write("\n")
            with open("../methods/GPT/data/{}/test_labels.txt".format(args.dataset), "w") as f:
                for line in test_set[args.label_name]:
                    f.write(str(line))
                    f.write("\n")
            os.chdir("../methods/GPT")
            os.system(
                "CUDA_VISIBLE_DEVICES={} python score.py {} --model {} --split test --seed {}"
                .format(args.gpu, args.dataset, args.method, args.random_state))
        else:
            os.system("mkdir -p ../methods/GPT/data/{}".format(args.dataset))
            os.system("cp class_names.txt ../methods/GPT/data/{}/class_names.txt".format(args.dataset))
            os.system("cp prompt.txt ../methods/GPT/data/{}/prompt.txt".format(args.dataset))
            A = np.random.permutation(np.arange(len(train_set[args.text_name])))
            n_shot = []
            label_num = [0 for _ in range(max(train_set[args.label_name])+1)]
            for i in range(len(train_set[args.text_name])):
                id = A[i]
                if label_num[train_set[args.label_name][id]] < args.n_shot:
                    n_shot.append(id)
                    label_num[train_set[args.label_name][id]] += 1
                if len(n_shot) == args.n_shot * len(label_num):
                    break
            with open("../methods/GPT/data/{}/n_shot.txt".format(args.dataset), "w") as f:
                for id in n_shot:
                    f.write(str(train_set[args.text_name][id]))
                    f.write("\n")
            with open("../methods/GPT/data/{}/n_shot_labels.txt".format(args.dataset), "w") as f:
                for id in n_shot:
                    f.write(str(train_set[args.label_name][id]))
                    f.write("\n")
            with open("../methods/GPT/data/{}/test.txt".format(args.dataset), "w") as f:
                for line in test_set[args.text_name]:
                    f.write(str(line))
                    f.write("\n")
            with open("../methods/GPT/data/{}/test_labels.txt".format(args.dataset), "w") as f:
                for line in test_set[args.label_name]:
                    f.write(str(line))
                    f.write("\n")
            os.chdir("../methods/GPT")
            os.system(
                "CUDA_VISIBLE_DEVICES={} python score.py {} --model {} --split test --n_shot {} --seed {}"
                .format(args.gpu, args.dataset, args.method, args.n_shot, args.random_state))
    elif args.method.startswith("gpt") and args.additional_method == "ProtoCal":
        assert args.prompt == True
        assert args.class_names == True
        assert args.seed_words == False
        os.system("mkdir -p ../methods/GPT/data/{}".format(args.dataset))
        os.system("cp class_names.txt ../methods/GPT/data/{}/class_names.txt".format(args.dataset))
        os.system("cp prompt.txt ../methods/GPT/data/{}/prompt.txt".format(args.dataset))
        with open("../methods/GPT/data/{}/train.txt".format(args.dataset), "w") as f:
            for line in train_set[args.text_name]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/GPT/data/{}/train_labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/GPT/data/{}/test.txt".format(args.dataset), "w") as f:
            for line in test_set[args.text_name]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/GPT/data/{}/test_labels.txt".format(args.dataset), "w") as f:
            for line in test_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.chdir("../methods/GPT")
        os.system(
            "CUDA_VISIBLE_DEVICES={} python ProtoCal.py {} --model {} --split test --seed {}"
            .format(args.gpu, args.dataset, args.method, args.random_state))


import os
import pickle
import json
import string

import pandas as pd
import numpy as np
from datasets import load_dataset, list_datasets, Dataset
import csv

DATA_FOLDER_PATH = os.path.join('..', 'data')

def split_data(args):
    if args.dataset in list_datasets():
        dataset = load_dataset(args.dataset, revision="plain_text", split="train", cache_dir='/data/tianle/huggingfacedatasetscache')
        if args.train_size < 1.0:
            train_test = dataset.train_test_split(train_size=args.train_size,
                                                  shuffle=True, seed=args.random_state)
            train_set = train_test["train"]
        else:
            train_set = dataset
        dataset = load_dataset(args.dataset, revision="plain_text", split=args.split, cache_dir='/data/tianle/huggingfacedatasetscache')
        if args.test_size < 1.0:
            train_test = dataset.train_test_split(test_size=args.test_size,
                                                  shuffle=True, seed=args.random_state)
            test_set = train_test["test"]
        else:
            test_set = dataset
        #test_set = train_test["test"]
    else:
        assert args.dataset in ['20News', 'NYT', 'NYT-Locations', 'NYT-Topics', '20News-fine', 'NYT-fine']
        dir = os.path.join(DATA_FOLDER_PATH, args.dataset, 'dataset.txt')
        with open(dir, mode='r', encoding='utf-8') as text_file:
            text_data = list(map(lambda x: x.strip(), text_file.readlines()))
        dir = os.path.join(DATA_FOLDER_PATH, args.dataset, 'labels.txt')
        with open(dir, mode='r', encoding='utf-8') as label_file:
            label_data = list(map(lambda x: int(x.strip()), label_file.readlines()))
        data = {
            "text": text_data,
            "label": label_data,
        }
        dataset = Dataset.from_dict(data)
        #dataset = load_dataset('text', split="train", data_files={"text": os.path.join(DATA_FOLDER_PATH, args.dataset, 'dataset.txt'), "label": os.path.join(DATA_FOLDER_PATH, args.dataset, 'labels.txt')})
        train_test_split = dataset.train_test_split(train_size=args.split_ratio, shuffle=True, seed=args.random_state)
        dataset = train_test_split["train"]
        if args.train_size < 1.0:
            train_test = dataset.train_test_split(train_size=args.train_size,
                                                  shuffle=True, seed=args.random_state)
            train_set = train_test["train"]
        else:
            train_set = dataset
        dataset = train_test_split["test"]
        if args.test_size < 1.0:
            train_test = dataset.train_test_split(test_size=args.test_size,
                                                  shuffle=True, seed=args.random_state)
            test_set = train_test["test"]
        else:
            test_set = dataset

    import string
    punctuation =  string.punctuation
    def concat(example):
        example["x-TC"] = ""
        for arg in args.text_name:
            example["x-TC"] += example[arg].strip()
            if example["x-TC"][len(example["x-TC"])-1] not in punctuation:
                example["x-TC"] += '.'
            example["x-TC"] += ' '
        example["x-TC"] = example["x-TC"].strip()
        return example
    train_set = train_set.map(concat)
    test_set = test_set.map(concat)
    print("Finish data split!")
    print("Sampled instance:", train_set[0])
    print("train size: {}, test size: {}".format(len(train_set), len(test_set)))

    num = {}
    for i in range(len(test_set[args.label_name])):
        label = test_set[args.label_name][i]
        if num.get(label):
            num[label] += 1
        else:
            num[label] = 1
    imbalance = num[max(num, key=num.get)] / num[min(num, key=num.get)]
    print("imbalance ratio: {}".format(imbalance))

    return train_set, test_set


def run_method(args, train_set, test_set):
    if args.method == "X-Class":
        assert args.class_names
        assert args.seed_words == False
        os.system("mkdir -p ../methods/X-Class/data/datasets/{}".format(args.dataset))
        os.system("cp {} ../methods/X-Class/data/datasets/{}/classes.txt".format(args.class_names_file, args.dataset))
        with open("../methods/X-Class/data/datasets/{}/dataset.txt".format(args.dataset), "w") as f:
            for line in train_set["x-TC"]:
                f.write(line)
                f.write("\n")
        with open("../methods/X-Class/data/datasets/{}/labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.system("mkdir -p ../methods/X-Class/data/datasets/{}_test".format(args.dataset))
        os.system("cp class_names2.txt ../methods/X-Class/data/datasets/{}_test/classes.txt".format(args.dataset))
        with open("../methods/X-Class/data/datasets/{}_test/dataset.txt".format(args.dataset), "w") as f:
            for line in test_set["x-TC"]:
                f.write(line)
                f.write("\n")
        with open("../methods/X-Class/data/datasets/{}_test/labels.txt".format(args.dataset), "w") as f:
            for line in test_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        if args.prompt == True:
            os.system("cp prompt.txt ../methods/X-Class/data/datasets/{}/prompt.txt".format(args.dataset))
            os.chdir("../methods/X-Class/scripts")
            os.system("chmod -R 777 run_prompt.sh")
            os.system("chmod -R 777 run_train_text_classifier.sh")
            if args.suffix == "":
                args.suffix = "64 bbu 12 bbc 128 mixture"  # "100 blu 24 blc 128 none"
            os.system("./run_prompt.sh {} {} {} {}".format(args.gpu, args.dataset, args.random_state, args.suffix))
        else:
            os.chdir("../methods/X-Class/scripts")
            os.system("chmod -R 777 run.sh")
            os.system("chmod -R 777 run_train_text_classifier.sh")
            if args.suffix == "":
                args.suffix = "64 bbu 12 bbc 128"  # "100 blu 24 blc 128"
            os.system("./run.sh {} {} {} {}".format(args.gpu, args.dataset, args.random_state, args.suffix))
    elif args.method == "BERTprompt":
        assert args.class_names
        assert args.seed_words==False
        assert args.prompt
        if args.additional_method == "ProtoCal":
            ####################

            # please check class2/prompt2.txt!!!!!

            ####################
            os.system("mkdir -p ../methods/X-Class/data/datasets/{}".format(args.dataset))
            os.system("cp class_names_BERT.txt ../methods/X-Class/data/datasets/{}/classes.txt".format(args.dataset))
            with open("../methods/X-Class/data/datasets/{}/dataset.txt".format(args.dataset), "w") as f:
                for line in train_set["x-TC"]:
                    f.write(line)
                    f.write("\n")
            with open("../methods/X-Class/data/datasets/{}/labels.txt".format(args.dataset), "w") as f:
                for line in train_set[args.label_name]:
                    f.write(str(line))
                    f.write("\n")
            os.system("mkdir -p ../methods/X-Class/data/datasets/{}_test".format(args.dataset))
            with open("../methods/X-Class/data/datasets/{}_test/dataset.txt".format(args.dataset), "w") as f:
                for line in test_set["x-TC"]:
                    f.write(line)
                    f.write("\n")
            with open("../methods/X-Class/data/datasets/{}_test/labels.txt".format(args.dataset), "w") as f:
                for line in test_set[args.label_name]:
                    f.write(str(line))
                    f.write("\n")
            os.system("cp prompt_BERT.txt ../methods/X-Class/data/datasets/{}/prompt.txt".format(args.dataset))
            os.chdir("../methods/X-Class/scripts")
            os.system("CUDA_VISIBLE_DEVICES={} python BERTprompt_ProtoCal.py --dataset_name {} --random_state {} {}".
                      format(args.gpu, args.dataset, args.random_state, args.suffix))
        else:
            os.system("mkdir -p ../methods/X-Class/data/datasets/{}_test".format(args.dataset))
            os.system("cp class_names_BERT.txt ../methods/X-Class/data/datasets/{}_test/classes.txt".format(args.dataset))
            with open("../methods/X-Class/data/datasets/{}_test/dataset.txt".format(args.dataset), "w") as f:
                for line in test_set["x-TC"]:
                    f.write(line)
                    f.write("\n")
            with open("../methods/X-Class/data/datasets/{}_test/labels.txt".format(args.dataset), "w") as f:
                for line in test_set[args.label_name]:
                    f.write(str(line))
                    f.write("\n")
            os.system("cp prompt_BERT.txt ../methods/X-Class/data/datasets/{}_test/prompt.txt".format(args.dataset))
            os.chdir("../methods/X-Class/scripts")
            os.system("CUDA_VISIBLE_DEVICES={} python BERTprompt.py --dataset_name {}_test --random_state {} {}".
                      format(args.gpu, args.dataset, args.random_state, args.suffix))
    elif args.method == "ConWea":
        assert args.class_names==False
        assert args.seed_words==True
        os.system("mkdir -p ../methods/ConWea/data/{}".format(args.dataset))
        label = [str(i) for i in train_set[args.label_name]]
        train={
            "sentence": train_set["x-TC"],
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
            "sentence": test_set["x-TC"],
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
        os.system("cp {} ../methods/LOTClass/datasets/{}/label_names.txt".format(args.class_names_file, args.dataset))
        with open("../methods/LOTClass/datasets/{}/train.txt".format(args.dataset), "w") as f:
            for line in train_set["x-TC"]:
                f.write(line)
                f.write("\n")
        with open("../methods/LOTClass/datasets/{}/train_labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/LOTClass/datasets/{}/test.txt".format(args.dataset), "w") as f:
            for line in test_set["x-TC"]:
                f.write(line)
                f.write("\n")
        with open("../methods/LOTClass/datasets/{}/test_labels.txt".format(args.dataset), "w") as f:
            for line in test_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.chdir("../methods/LOTClass")
        os.system("CUDA_VISIBLE_DEVICES={} python src/train.py --dataset_dir datasets/{}/ --test_file test.txt --test_label_file test_labels.txt --train_batch_size 32 --accum_steps 4 --gpus 1 --random_state {} {}"
                  .format(args.gpu, args.dataset, args.random_state, args.suffix))
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
                "sentence": train_set["x-TC"]
            }
            df = pd.DataFrame(train)
            df.to_csv("../methods/WeSTClass/{}/dataset.csv".format(args.dataset), header=False, index=False)
            test = {
                "label": test_set[args.label_name],
                "sentence": test_set["x-TC"]
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
                "sentence": train_set["x-TC"]
            }
            df = pd.DataFrame(train)
            df.to_csv("../methods/WeSTClass/{}/dataset.csv".format(args.dataset), header=False, index=False)
            test = {
                "label": test_set[args.label_name],
                "sentence": test_set["x-TC"]
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
        os.system("mkdir -p ../methods/ClassKG/data/processed/{}".format(args.dataset))
        if args.seed_words == True:
            assert args.class_names == False
            with open("seed_words.txt", mode='r', encoding='utf-8') as f:
                seeds = list(map(lambda x: x.strip(), f.readlines()))
        elif args.class_names == True:
            assert args.seed_words == False
            with open(args.class_names_file, mode='r', encoding='utf-8') as f:
            #with open("class_names.txt", mode='r', encoding='utf-8') as f:
                seeds = list(map(lambda x: x.strip(), f.readlines()))
        seed_words = {}
        for i in range(len(seeds)):
            seed_words[i] = seeds[i].split(' ')
        with open("../methods/ClassKG/data/processed/{}/keywords.json".format(args.dataset), 'w') as f:
            json.dump(seed_words, f, indent=2)

        label = [str(i) for i in test_set[args.label_name]]
        test = {
            "sentence": test_set["x-TC"],
            "label": label
        }

        train_list = []
        for i in range(len(train_set["x-TC"])):
            train_list.append([train_set[i]["x-TC"], train_set[i][args.label_name]])
        test_list = []
        for i in range(len(test_set["x-TC"])):
            test_list.append([test_set[i]["x-TC"], test_set[i][args.label_name]])
        with open("../methods/ClassKG/data/processed/{}/unlabeled.json".format(args.dataset), 'w') as f:
            json.dump(train_list, f, indent=2)
        with open("../methods/ClassKG/data/processed/{}/test.json".format(args.dataset), 'w') as f:
            json.dump(test_list, f, indent=2)

        import yaml
        default_para = {
            "data_dir_name": args.dataset,

            "model": {
                "number_classes": len(seeds)
            },

            "SSL": {
                "enable": True,
                "number_itr": 100000
            },

            "keywords_update": {
                "extract_keywords_per_class": [1000],
                "keywords_set_keep_max_num": [1000],
                "IDF_n": 4,
                "overwrite_conflict": False
            },

            "classifier": {
                "n_epochs": 100,
                "batch_size": 8,
                "stop_itr": [400],
                "lr": 5e-6,
                "eval_interval": 80,
                "type": "short"
            },

            "trainer_Graph": {
                "epoch": [4, 10],
                "batch_size": 256
            },

            "file_path": {
                "save_dir": args.dataset
            },
        }
        with open("../methods/ClassKG/config/{}.yaml".format(args.dataset), 'w') as f:
            yaml.dump(data=default_para, stream=f, allow_unicode=True, default_flow_style=None)

        os.chdir("../methods/ClassKG/task")
        os.system("python pipeline.py --gpu {} --dataset {} --random_state {}".format(args.gpu, args.dataset, args.random_state))
    elif args.method.startswith("gpt") and args.additional_method == None:
        assert args.prompt == True
        assert args.class_names == True
        assert args.seed_words == False
        if args.n_shot == 0:
            os.system("mkdir -p ../methods/GPT/data/{}".format(args.dataset))
            os.system("cp {} ../methods/GPT/data/{}/class_names.txt".format(args.class_names_file, args.dataset))
            #os.system("cp class_names_gpt.txt ../methods/GPT/data/{}/class_names.txt".format(args.dataset))
            os.system("cp {} ../methods/GPT/data/{}/prompt.txt".format(args.prompt_file, args.dataset))
            with open("../methods/GPT/data/{}/train.txt".format(args.dataset), "w") as f:
                for line in train_set["x-TC"]:
                    f.write(str(line))
                    f.write("\n")
            with open("../methods/GPT/data/{}/train_labels.txt".format(args.dataset), "w") as f:
                for line in train_set[args.label_name]:
                    f.write(str(line))
                    f.write("\n")
            with open("../methods/GPT/data/{}/test.txt".format(args.dataset), "w") as f:
                 for line in test_set["x-TC"]:
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
            os.system("cp {} ../methods/GPT/data/{}/class_names.txt".format(args.class_names_file, args.dataset))
            #os.system("cp class_names_gpt.txt ../methods/GPT/data/{}/class_names.txt".format(args.dataset))
            os.system("cp {} ../methods/GPT/data/{}/prompt.txt".format(args.prompt_file, args.dataset))
            A = np.random.permutation(np.arange(len(train_set["x-TC"])))
            n_shot = []
            label_num = [0 for _ in range(max(train_set[args.label_name])+1)]
            for i in range(len(train_set["x-TC"])):
                id = A[i]
                if label_num[train_set[args.label_name][id]] < args.n_shot:
                    n_shot.append(id)
                    label_num[train_set[args.label_name][id]] += 1
                if len(n_shot) == args.n_shot * len(label_num):
                    break
            with open("../methods/GPT/data/{}/n_shot.txt".format(args.dataset), "w") as f:
                for id in n_shot:
                    f.write(str(train_set["x-TC"][id]))
                    f.write("\n")
            with open("../methods/GPT/data/{}/n_shot_labels.txt".format(args.dataset), "w") as f:
                for id in n_shot:
                    f.write(str(train_set[args.label_name][id]))
                    f.write("\n")
            with open("../methods/GPT/data/{}/test.txt".format(args.dataset), "w") as f:
                for line in test_set["x-TC"]:
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
        os.system("cp {} ../methods/GPT/data/{}/class_names.txt".format(args.class_names_file, args.dataset))
        #os.system("cp class_names_gpt.txt ../methods/GPT/data/{}/class_names.txt".format(args.dataset))
        os.system("cp {} ../methods/GPT/data/{}/prompt.txt".format(args.prompt_file, args.dataset))
        with open("../methods/GPT/data/{}/train.txt".format(args.dataset), "w") as f:
            for line in train_set["x-TC"]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/GPT/data/{}/train_labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/GPT/data/{}/test.txt".format(args.dataset), "w") as f:
            for line in test_set["x-TC"]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/GPT/data/{}/test_labels.txt".format(args.dataset), "w") as f:
            for line in test_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.chdir("../methods/GPT")
        os.system(
            "CUDA_VISIBLE_DEVICES={} python ProtoCal.py {} --model {} --split test --seed {} {}"
            .format(args.gpu, args.dataset, args.method, args.random_state, args.suffix))
    elif args.method.startswith("NPPrompt"):
        assert args.prompt == True
        assert args.class_names == True
        assert args.seed_words == False
        os.system("mkdir -p ../methods/NPPrompt/datasets/{}".format(args.dataset))
        os.system("cp class_names_NPP.txt ../methods/NPPrompt/datasets/{}/class_names.txt".format(args.dataset))
        os.system("cp prompt_NPP.txt ../methods/NPPrompt/datasets/{}/prompt.txt".format(args.dataset))
        with open("../methods/NPPrompt/datasets/{}/train.txt".format(args.dataset), "w") as f:
            for line in train_set["x-TC"]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/NPPrompt/datasets/{}/train_labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/NPPrompt/datasets/{}/test.txt".format(args.dataset), "w") as f:
            for line in test_set["x-TC"]:
                f.write(str(line))
                f.write("\n")
        with open("../methods/NPPrompt/datasets/{}/test_labels.txt".format(args.dataset), "w") as f:
            for line in test_set[args.label_name]:
                f.write(str(line))
                f.write("\n")
        os.chdir("../methods/NPPrompt")
        os.system("sh example_run.sh {} {} {} {}".format(args.gpu, args.dataset, args.random_state, args.suffix))


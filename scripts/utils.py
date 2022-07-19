import os
from datasets import load_dataset, list_datasets

def split_data(args):
    if args.dataset in list_datasets():
        train_set, test_set = load_dataset(args.dataset)
        print(len(train_set),len(test_set))
        train_set, test_set = dataset.train_test_split(test_size=args.test_size, train_size=args.train_size,
                                 shuffle=True, random_seed=args.random_seed)
    else:
        ...
    print("Finish data split!")
    return train_set, test_set


def run_method(args, train_set, test_set):
    if args.method == "XClass":
        assert args.class_names
        assert args.seed_words==False
        os.mkdir("../methods/X-Class/data/datasets/{}".format(args.dataset))
        os.system("cp class_names.txt ../methods/X-Class/data/datasets/{}/classes.txt".format(args.dataset))
        with open("../methods/X-Class/data/datasets/{}/dataset.txt".format(args.dataset), "w") as f:
            for line in train_set["text"]:
                f.write(line)
                f.write("\n")
        with open("../methods/X-Class/data/datasets/{}/labels.txt".format(args.dataset), "w") as f:
            for line in train_set[args.label_name]:
                f.write(line)
                f.write("\n")
        os.mkdir("../methods/X-Class/data/datasets/{}_test".format(args.dataset))
        os.system("cp class_names.txt ../methods/X-Class/data/datasets/{}_test/classes.txt".format(args.dataset))
        with open("../methods/X-Class/data/datasets/{}_test/dataset.txt".format(args.dataset), "w") as f:
            for line in test_set["text"]:
                f.write(line)
                f.write("\n")
        with open("../methods/X-Class/data/datasets/{}_test/labels.txt".format(args.dataset), "w") as f:
            for line in test_set[args.label_name]:
                f.write(line)
                f.write("\n")
        os.system("./ ../methods/X-Class/scripts/run.sh {} {}".format(args.gpu, args.dataset))
    elif args.method == "ConWea":
        assert args.class_names==False
        assert args.seed_words==True
        os.mkdir("../methods/ConWea/data/{}".format(args.dataset))
        os.mkdir("../methods/ConWea/data/{}".format(args.dataset + "_intermediate"))
        ...
        ...
        ...
        os.system("python ../methods/ConWea/contextualize.py --dataset_path data/{} --temp_dir data/{} --gpu_id {}"
                  .format(args.dataset, args.dataset + "_intermediate", args.gpu))
        os.system("python ../methods/ConWea/train.py --dataset_path data/{} --gpu_id {}"
                  .format(args.dataset, args.gpu))
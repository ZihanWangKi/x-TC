import os
import argparse
from datasets import load_dataset, list_datasets, Dataset

# please make sure no dir or file with the same huggingface dataset name in this dir
# only support huggingface datasets
# please add label_names.txt, prompt.txt (optional) manually

def split(args):
    assert args.dataset in list_datasets()
    dataset = load_dataset(args.dataset, split="train")
    if args.train_size < 1.0:
        train_test = dataset.train_test_split(train_size=args.train_size,
                                              shuffle=True, seed=args.random_state)
        train_set = train_test["train"]
    else:
        train_set = dataset
    dataset = load_dataset(args.dataset, split=args.split)
    if args.test_size < 1.0:
        train_test = dataset.train_test_split(test_size=args.test_size,
                                              shuffle=True, seed=args.random_state)
        test_set = train_test["test"]
    else:
        test_set = dataset

    import string
    punctuation = string.punctuation
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
    print("sample #0:", train_set[0])
    print("train size: {}, test size: {}".format(len(train_set), len(test_set)))

    # report imbalance ratio
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="ag_news",
                        help="huggingface dataset name")
    parser.add_argument("--text_name", default=["text"], nargs='+',
                        help="The exact text names in huggingface dataset, processor will concat them together."
                             "e.g. ['title', 'content'] for dbpedia_14.")
    parser.add_argument("--label_name", default="label",
                        help="The exact label name in huggingface dataset, 'label' for most cases.")
    parser.add_argument("--split", default="test",
                        help="test on test/validation/... split, please check the huggingface dataset.")
    parser.add_argument("--train_size", type=float, default=1.0,
                        help="The ratio of the train set.")
    parser.add_argument("--test_size", type=float, default=1.0,
                       help="The ratio of the test set.")
    parser.add_argument("--train_label", action="store_true", default=False,
                        help="Control whether train_label.txt is provided.")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    print(vars(args))

    train_set, test_set = split(args)

    os.system(f"mkdir -p ./{args.dataset}")

    with open(f"./{args.dataset}/train_text.txt", "w") as f:
        for line in train_set["x-TC"]:
            f.write(str(line))
            f.write("\n")

    if args.train_label:
        with open(f"./{args.dataset}/train_label.txt", "w") as f:
            for line in train_set[args.label_name]:
                f.write(str(line))
                f.write("\n")

    with open(f"./{args.dataset}/test_text.txt", "w") as f:
        for line in test_set["x-TC"]:
            f.write(str(line))
            f.write("\n")

    with open(f"./{args.dataset}/test_label.txt", "w") as f:
        for line in test_set[args.label_name]:
            f.write(str(line))
            f.write("\n")

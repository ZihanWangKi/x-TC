import argparse
from utils import *


def main(args):
    train_set, test_set = split_data(args)
    run_method(args, train_set, test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="ag_news", help="dataset name")
    parser.add_argument("--method", default="X-Class", help="method name")
    parser.add_argument("--label_name", default="label",
                        help="The exact name of the label in raw data, e.g. 'label-coarse'.")
    parser.add_argument("--train_size", type=float, default=0.6,
                        help="The ratio of the split train set (unlabeled by default)")
    parser.add_argument("--test_size", type=float, default=0.4,
                        help="The ratio of the split test set. Make sure test_ratio + train ratio <= 1")
    # parser.add_argument("--labeled_ratio", default=0.0, help="extra parameter for few-labeled data methods")
    parser.add_argument("--class_names", type=bool, default=True,
                        help="Set to False if the method doesn't need class names.")
    parser.add_argument("--seed_words", type=bool, default=False,
                        help="Set to True if you want to test seed-words based methods. xxx") # todo
    parser.add_argument("--gpu", default=0, help="gpu id")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    assert args.train_size + args.test_size <= 1.0
    print(vars(args))

    main(args)

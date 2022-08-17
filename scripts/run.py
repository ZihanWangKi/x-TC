import argparse
import random
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
    parser.add_argument("--train_size", type=float, default=1.0,
                        help="The ratio of the split train set (unlabeled by default)")
    # parser.add_argument("--labeled_ratio", default=0.0, help="extra parameter for few-labeled data methods")
    parser.add_argument("--class_names", action="store_true",
                        help="Set to True if you want to test class-names based methods."
                             "Please enter the class names into class_names.txt in the current directory."
                             "One class name per line, and the order corresponds to the label order.")
    parser.add_argument("--seed_words", action="store_true",
                        help="Set to True if you want to test seed-words based methods."
                             "Please enter the seed words into seed_words.txt in the current directory."
                             "Enter the seed words belonging to the same class on the same line, separated by spaces,"
                             "and the order of lines corresponds to the label order.")
    parser.add_argument("--gpu", default=0, help="gpu id")
    parser.add_argument("--random_state", type=int, default=42) # todo

    random.seed(args.random_state)
    args = parser.parse_args()
    print(vars(args))

    main(args)

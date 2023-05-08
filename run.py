import argparse
import random
import numpy
from utils import Dataset, Labels, data_process, get_method, evaluate_predictions

DATA_CHOICES = ['agnews_sampled']
METHOD_CHOICES = ['xclass']

def apply(dataset_name, train_dataset, train_label, method):
    method.apply(dataset_name, train_dataset, train_label)

def infernece(dataset_name, test_dataset, method):
    method.inference(dataset_name, test_dataset)

def evaluate(dataset_name, test_label, method):
    # load predictions from the saved file and calc F1 score
    preds = method.load_pred(dataset_name)
    evaluate_predictions(test_label.labels, preds)

parser = argparse.ArgumentParser(description='x-TC')
parser.add_argument('--data', type=str, default='agnews_sampled', choices=DATA_CHOICES)
parser.add_argument('--method', type=str, default='xclass', choices=METHOD_CHOICES)
args = parser.parse_args()

# dataset loading
train_dataset, train_label, test_dataset, test_label = data_process(args.data)

method = get_method(args.method)

# apply method
apply(args.data, train_dataset, train_label, method)
infernece(args.data, test_dataset, method)
evaluate(args.data, test_label, method)

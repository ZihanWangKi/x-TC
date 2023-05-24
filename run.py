import argparse
import random
import numpy
from utils import Dataset, Labels, data_process, get_method, evaluate_predictions

DATA_CHOICES = ['agnews_sampled', '20News', '20News-fine', 'NYT', 'NYT-fine',
                'NYT-Locations', 'NYT-Topics', 'ag_news', 'dbpedia_14', 'imdb',
                'yelp_polarity', 'yelp_review_full']
METHOD_CHOICES = ['xclass', 'prompt', 'prompt_gpt', 'lotclass']

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
parser.add_argument('--base_model', type=str, required=True, help='pre-traiend model to use, e.g., bert-base-uncased')
parser.add_argument('--label_names_file_name', type=str, default='data/agnews_sampled/label_names.txt', help='file name of label names') # we can add the variants.
parser.add_argument('--prompt_file_name', type=str, default='data/agnews_sampled/prompt.txt', help='file name of prompt')
parser.add_argument('--hyperparameter_file_path', type=str, default='methods/hyperparameters/xclass.json', help='file name of hyperparameter')
args = parser.parse_args()

# dataset loading
train_dataset, train_label, test_dataset, test_label = data_process(args.data, args.label_names_file_name, args.prompt_file_name)

method = get_method(args.method, args.hyperparameter_file_path, args.base_model)

# apply method
apply(args.data, train_dataset, train_label, method)
infernece(args.data, test_dataset, method)
evaluate(args.data, test_label, method)

import argparse
from utils import Dataset, Labels

DATA_CHOICES = ['agnews_sampled']
METHOD_CHOICES = ['xclass']

def apply(train_dataset, method):
    
    return xxx

def infernece(test_dataset, method, xxx):
    
    return xxx

def evaluate(test_labels, xxx):
    xxx

parser = argparse.ArgumentParser(description='x-TC')
parser.add_argument('--data', type=str, default='agnews_sampled', choices=DATA_CHOICES)
parser.add_argument('--method', type=str, default='xclass', choices=METHOD_CHOICES)
args = parser.parse_args()

# dataset loading
train_dataset = xxx
test_dataset = xxx
test_labels = xxx

# apply method


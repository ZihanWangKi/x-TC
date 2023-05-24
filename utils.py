import os
import json
from typing import List, Optional
from sklearn.metrics import confusion_matrix, f1_score

# this file stores all utility functions, such as calculating the metrics, preprocessing the datasets


class Dataset:
    def __init__(self, texts: List[str], label_names: List[str], prompt: Optional[str] = None):
        self.texts = texts
        self.label_names = label_names
        self.prompt = prompt


class Labels:
    def __init__(self, labels: List[int]):
        self.labels = labels

def data_process(dataset_name, label_name_dir, prompt_dir):
    train_text_path = os.path.join("data", dataset_name, "train_text.txt")
    with open(train_text_path, mode='r', encoding='utf-8') as file:
        train_text = list(map(lambda x: x.strip(), file.readlines()))

    label_names_path = label_name_dir #os.path.join("data", dataset_name, "label_names.txt")
    with open(label_names_path, mode='r', encoding='utf-8') as file:
        label_names = list(map(lambda x: x.strip(), file.readlines()))

    test_text_path = os.path.join("data", dataset_name, "test_text.txt")
    with open(test_text_path, mode='r', encoding='utf-8') as file:
        test_text = list(map(lambda x: x.strip(), file.readlines()))

    test_label_path = os.path.join("data", dataset_name, "test_label.txt")
    with open(test_label_path, mode='r', encoding='utf-8') as file:
        test_label = list(map(lambda x: int(x.strip()), file.readlines()))

    train_label_path = os.path.join("data", dataset_name, "train_label.txt")
    train_label = None
    if os.path.exists(train_label_path):
        with open(train_label_path, mode='r', encoding='utf-8') as file:
            train_label = list(map(lambda x: x.strip(), file.readlines()))

    prompt_path = prompt_dir # os.path.join("data", dataset_name, "prompt.txt")
    prompt = None
    if os.path.exists(prompt_path):
        with open(prompt_path, mode='r', encoding='utf-8') as file:
            prompt = file.read()

    train_Dataset = Dataset(train_text, label_names, prompt)
    train_Labels = Labels(train_label)
    test_Dataset = Dataset(test_text, label_names, prompt)
    test_Labels = Labels(test_label)

    return train_Dataset, train_Labels, test_Dataset, test_Labels

def get_method(method_name, hyperparameter_file_path, base_model):
    if method_name == "xclass":
        from methods.xclass_method import xclass, xclassHyperparams
        hyperparameters = xclassHyperparams.from_json(json.load(open(hyperparameter_file_path, mode='r')))
        method = xclass() # TODO, pass in hyperparameters and base_model
    elif method_name == "prompt":
        from methods.prompt_method import prompt, promptHyperparams
        hyperparameters = promptHyperparams.from_dict(json.load(open(hyperparameter_file_path, mode='r')))
        method = prompt(hyperparameters, base_model)
    elif method_name == "prompt_gpt":
        from methods.prompt_gpt_method import prompt_gpt, prompt_gptHyperparams
        hyperparameters = prompt_gptHyperparams.from_dict(json.load(open(hyperparameter_file_path, mode='r')))
        method = prompt_gpt(hyperparameters, base_model)
    elif method_name == "lotclass":
        from methods.lotclass_method import lotclass, lotclassHyperparams
        hyperparameters = lotclassHyperparams.from_dict(json.load(open(hyperparameter_file_path, mode='r')))
        method = lotclass(hyperparameters, base_model)
    elif method_name == "npprompt":
        from methods.npprompt_method import npprompt, nppromptHyperparams
        hyperparameters = nppromptHyperparams.from_dict(json.load(open(hyperparameter_file_path, mode='r')))
        method = npprompt(hyperparameters, base_model)
    else:
        ...

    return method


def evaluate_predictions(true_class, predicted_class, output_to_console=True, return_tuple=False):
    confusion = confusion_matrix(true_class, predicted_class)
    if output_to_console:
        print("-" * 80 + "Evaluating" + "-" * 80)
        print(confusion)
    f1_macro = f1_score(true_class, predicted_class, average='macro')
    f1_micro = f1_score(true_class, predicted_class, average='micro')
    if output_to_console:
        print("F1 macro: " + str(f1_macro))
        print("F1 micro: " + str(f1_micro))
    if return_tuple:
        return confusion, f1_macro, f1_micro
    else:
        return {
            "confusion": confusion.tolist(),
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }
import os
from typing import List, Optional
from sklearn.metrics import confusion_matrix, f1_score
from xclass_method import xclass

# this file stores all utility functions, such as calculating the metrics, preprocessing the datasets


class Dataset:
    def __init__(self, texts: List[str], label_names: List[str], prompt: Optional[str] = None):
        self.texts = texts
        self.label_names = label_names
        self.prompt = prompt
        
class Labels:
    def __init__(self, labels: List[int]):
        self.labels = labels

def data_process(dataset_name):
    train_text_path = os.path.join("data", dataset_name, "train_text.txt")
    with open(train_text_path, mode='r', encoding='utf-8') as file:
        train_text = list(map(lambda x: x.strip(), file.readlines()))

    label_names_path = os.path.join("data", dataset_name, "label_names.txt")
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

    prompt_path = os.path.join("data", dataset_name, "prompt.txt")
    prompt = None
    if os.path.exists(prompt_path):
        with open(prompt_path, mode='r', encoding='utf-8') as file:
            prompt = file.read()

    train_Dataset = Dataset(train_text, label_names, prompt)
    train_Labels = Labels(train_label)
    test_Dataset = Dataset(test_text, label_names, prompt)
    test_Labels = Labels(test_label)

    return train_Dataset, train_Labels, test_Dataset, test_Labels

def get_method(method_name):
    if method_name == "xclass":
        method = xclass()
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
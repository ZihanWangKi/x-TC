import os
import json
import yaml
from dataclasses import dataclass
from utils import Dataset, Labels
from dataclasses_json import dataclass_json
from typing import Optional

# support bert, roberta

# While the original classkg implementation supports multi-GPU execution,
# our benchmark strictly constrains all methods to run on a single GPU to
# ensure fairness.

@dataclass_json
@dataclass
class classkgHyperparams:
    total_iter: int = 3 # total epoches
    cluster: bool = False # whether to add clustering
    config_file_name: str = "methods/others/classkg.yaml"
    # config file required by classkg
    # Since there are too many changeable parameters, we only give our default setting.
    # If you want to further design your own config file, please refer to `external/classkg/config/defaults.py`.
    # This file defined all the changeable parameters.
    # Note that we will overwrite parameters 'data_dir_name', 'number_classes', 'save_dir'
    # in `apply()` or `inference()` since they are known or related to data saving.
    random_state: int = 42

class classkg():
    def __init__(self, hyperparams: classkgHyperparams, base_model):
        retval = os.getcwd()
        self.method_path = os.path.join(retval, "external", "classkg")
        self.hyperparams = hyperparams
        self.base_model = base_model
        with open(self.hyperparams.config_file_name, "r") as f:
            self.config = yaml.safe_load(f)
            print(self.config)

    def apply(self, dataset_name, train_dataset: Dataset, train_label: Labels):
        dataset_path = os.path.join(self.method_path, "data", "processed", dataset_name)
        os.system(f"rm -rf {dataset_path}")
        os.system(f"mkdir -p {dataset_path}")
        with open(f"{dataset_path}/keywords.json", "w") as f:
            json.dump({idx: [_] for idx, _ in enumerate(train_dataset.label_names)}, f, indent=2)

        train_list = []
        for i in range(len(train_dataset.texts)):
            train_list.append([train_dataset.texts[i], train_label.labels[i] if train_label.labels is not None else 0])
        # better provide train labels, classkg will report performance on train data during training.

        with open(f"{dataset_path}/unlabeled.json", "w") as f:
            json.dump(train_list, f, indent=2)

        para = {
            "data_dir_name": dataset_name,
            "model": {
                "number_classes": len(train_dataset.label_names)
            },
            "file_path": {
                "save_dir": dataset_path,
            },
        } # these parameters will be overwritten.
        self.config.update(para)

        with open(f"{self.method_path}/config/{dataset_name}.yaml", "w") as f:
            yaml.dump(data=self.config, stream=f, allow_unicode=True, default_flow_style=None)

        # train
        os.chdir(f"{self.method_path}/task")
        os.system(f"python pipeline.py --dataset {dataset_name} --random_state {self.hyperparams.random_state} "
                  f"--total_iter {self.hyperparams.total_iter} --lm {self.base_model} "
                  f"{'--cluster' if self.hyperparams.cluster else ''}")

    def inference(self, dataset_name, test_dataset: Dataset):
        inference_path = os.path.join(self.method_path, "data", "processed", dataset_name)
        os.system(f"mkdir -p {inference_path}")
        with open(f"{inference_path}/keywords.json", "w") as f:
            json.dump(test_dataset.label_names, f, indent=2)

        test_list = []
        for i in range(len(test_dataset.texts)):
            test_list.append([test_dataset.texts[i], 0])
        with open(f"{inference_path}/test.json", "w") as f:
            json.dump(test_list, f, indent=2)

        para = {
            "data_dir_name": dataset_name,
            "model": {
                "number_classes": len(test_dataset.label_names)
            },
            "file_path": {
                "save_dir": inference_path,
            },
        }  # these parameters will be overwritten.
        self.config.update(para)

        with open(f"{self.method_path}/config/{dataset_name}.yaml", "w") as f:
            yaml.dump(data=self.config, stream=f, allow_unicode=True, default_flow_style=None)

        # inference
        os.chdir(f"{self.method_path}/task")
        os.system(f"python pipeline.py --dataset {dataset_name} --random_state {self.hyperparams.random_state} "
                  f"--total_iter {self.hyperparams.total_iter} --lm {self.base_model} "
                  f"{'--cluster' if self.hyperparams.cluster else ''} --test_mode")

    def load_pred(self, dataset_name):
        output_dir = os.path.join(self.method_path, "inference", f"{dataset_name}")
        with open(os.path.join(output_dir, "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
        return pred_labels

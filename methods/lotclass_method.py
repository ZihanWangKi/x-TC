import os
import json
from dataclasses import dataclass
from datetime import datetime

from utils import Dataset, Labels
from dataclasses_json import dataclass_json
from typing import Optional

# support bert

# While the original lotclass implementation supports multi-GPU execution,
# our benchmark strictly constrains all methods to run on a single GPU to
# ensure fairness. But if you want to run lotclass on multiple GPUs, it is
# also easy to modify this file by adding related hyperparameters.


@dataclass_json
@dataclass
class lotclassHyperparams:
    eval_batch_size: int = 128 # batch size per GPU for evaluation
    train_batch_size: int = 32 # batch size per GPU for training
    top_pred_num: int = 100 # language model MLM top prediction cutoff
    category_vocab_size: int = 100 #category vocabulary size for each class
    match_threshold: int = 10 # category indicative words matching threshold
    max_len: int = 128 # length that documents are padded/truncated to
    update_interval: int = 50 # self training update interval; 50 is good in general
    accum_steps: int = 1 # gradient accumulation steps during training
    mcp_epochs: int = 3 # masked category prediction training epochs; 3-5 usually is good depending on dataset size (smaller dataset needs more epochs)
    self_train_epochs: int = 5 # self training epochs; 1-5 usually is good depending on dataset size (smaller dataset needs more epochs)
    early_stop: bool = False # whether to enable early stop of self-training
    random_state: int = 42

class lotclass():
    def __init__(self, hyperparams: lotclassHyperparams, base_model):
        retval = os.getcwd()
        self.method_path = os.path.join(retval, "external", "lotclass")
        self.hyperparams = hyperparams
        self.base_model = base_model
        exp = base_model.replace("/", "-") + f"/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        self.exp_name = os.path.join(retval, "experiment", "lotclass", exp)
        os.system(f"mkdir -p {self.exp_name}")

    def apply(self, dataset_name, train_dataset: Dataset, train_label: Labels):
        dataset_path = os.path.join(self.exp_name, "datasets", dataset_name)
        os.system(f"rm -rf {dataset_path}")
        os.system(f"mkdir -p {dataset_path}")
        with open(f"{dataset_path}/label_names.txt", "w") as f:
            for line in train_dataset.label_names:
                f.write(line)
                f.write("\n")
        with open(f"{dataset_path}/train.txt", "w") as f:
            for line in train_dataset.texts:
                f.write(line)
                f.write("\n")
        if train_label.labels is not None:
            with open(f"{dataset_path}/train_labels.txt", "w") as f:
                for line in train_label.labels:
                    f.write(str(line))
                    f.write("\n")

        # train
        os.chdir(f"{self.method_path}")
        os.system(f"python src/train.py --dataset_dir {dataset_path} --gpus 1 "
                  f"--lm_type {self.base_model}  --random_state {self.hyperparams.random_state} "
                  f"--eval_batch_size {self.hyperparams.eval_batch_size} "
                  f"--train_batch_size {self.hyperparams.train_batch_size} "
                  f"--top_pred_num {self.hyperparams.top_pred_num} "
                  f"--category_vocab_size {self.hyperparams.category_vocab_size} "
                  f"--match_threshold {self.hyperparams.match_threshold} "
                  f"--max_len {self.hyperparams.max_len} "
                  f"--update_interval {self.hyperparams.update_interval} "
                  f"--accum_steps {self.hyperparams.accum_steps} "
                  f"--mcp_epochs {self.hyperparams.mcp_epochs} "
                  f"--self_train_epochs {self.hyperparams.self_train_epochs} "
                  f"{' ' if self.hyperparams.early_stop else '--early_stop'}")

    def inference(self, dataset_name, test_dataset: Dataset):
        inference_path = os.path.join(self.exp_name, "datasets", dataset_name)
        os.system(f"mkdir -p {inference_path}")
        with open(f"{inference_path}/label_names.txt", "w") as f:
            for line in test_dataset.label_names:
                f.write(line)
                f.write("\n")
        with open(f"{inference_path}/test.txt", "w") as f:
            for line in test_dataset.texts:
                f.write(line)
                f.write("\n")

        # inference
        os.chdir(f"{self.method_path}")
        os.system(f"python src/train.py --dataset_dir {inference_path} "
                  f"--test_file test.txt --gpus 1 "
                  f"--lm_type {self.base_model}  --random_state {self.hyperparams.random_state} "
                  f"--eval_batch_size {self.hyperparams.eval_batch_size} "
                  f"--train_batch_size {self.hyperparams.train_batch_size} "
                  f"--top_pred_num {self.hyperparams.top_pred_num} "
                  f"--category_vocab_size {self.hyperparams.category_vocab_size} "
                  f"--match_threshold {self.hyperparams.match_threshold} "
                  f"--max_len {self.hyperparams.max_len} "
                  f"--update_interval {self.hyperparams.update_interval} "
                  f"--accum_steps {self.hyperparams.accum_steps} "
                  f"--mcp_epochs {self.hyperparams.mcp_epochs} "
                  f"--self_train_epochs {self.hyperparams.self_train_epochs} "
                  f"{' ' if self.hyperparams.early_stop else '--early_stop'}")

    def load_pred(self, dataset_name):
        output_dir = os.path.join(self.exp_name, "inference", f"{dataset_name}")
        with open(os.path.join(output_dir, "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
        return pred_labels

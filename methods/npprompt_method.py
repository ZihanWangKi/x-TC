import os
import json
from dataclasses import dataclass
from utils import Dataset, Labels
from dataclasses_json import dataclass_json
from typing import Optional

# support bert (-cased is better), roberta (can be expanded)

# if you choose to customize your own prompt, please ensure that it contains
# only a single {} bracket for the input text (like prompt and prompt_gpt).
# We will then adapt the prompt to npprompt.

@dataclass_json
@dataclass
class nppromptHyperparams:
    select: int = 100 # number of selected keywords for each classes
    cluster: bool = False # whether to add clustering
    prompt: Optional[str] = None # you can overwrite the prompt here
    random_state: int = 42

class npprompt():
    def __init__(self, hyperparams: nppromptHyperparams, base_model):
        retval = os.getcwd()
        self.method_path = os.path.join(retval, "external", "npprompt")
        self.hyperparams = hyperparams
        self.base_model = base_model

    def apply(self, dataset_name, train_dataset: Dataset, train_label: Labels):
        # apply() only for prototypical calibration

        if self.hyperparams.cluster:
            assert train_dataset.prompt is not None or self.hyperparams.prompt is not None
            dataset_path = os.path.join(self.method_path, "datasets", dataset_name)
            os.system(f"mkdir -p {dataset_path}")
            with open(f"{dataset_path}/class_names.txt", "w") as f:
                for line in train_dataset.label_names:
                    f.write(line)
                    f.write("\n")
            with open(f"{dataset_path}/train.txt", "w") as f:
                for line in train_dataset.texts:
                    f.write(line)
                    f.write("\n")
            prompt = self.hyperparams.prompt if self.hyperparams.prompt is not None else train_dataset.prompt
            prompt = prompt.strip().format("{\"placeholder\": \"text_a\"}") + " {\"mask\"}"
            with open(f"{dataset_path}/prompt.txt", "w") as f:
                f.write(prompt)

            model = self.base_model.split('-')[0]

            # train
            os.chdir(f"{self.method_path}")
            os.system(f"python emb_prompt_cluster.py --dataset {dataset_name} --model {model} --model_name_or_path {self.base_model} "
                      f"--select {self.hyperparams.select} --seed {self.hyperparams.random_state} ")

    def inference(self, dataset_name, test_dataset: Dataset):
        inference_path = os.path.join(self.method_path, "datasets", dataset_name)
        os.system(f"mkdir -p {inference_path}")
        assert test_dataset.prompt is not None or self.hyperparams.prompt is not None

        with open(f"{inference_path}/class_names.txt", "w") as f:
            for line in test_dataset.label_names:
                f.write(line)
                f.write("\n")
        with open(f"{inference_path}/test.txt", "w") as f:
            for line in test_dataset.texts:
                f.write(line)
                f.write("\n")
        prompt = self.hyperparams.prompt if self.hyperparams.prompt is not None else test_dataset.prompt
        prompt = prompt.strip().format("{\"placeholder\": \"text_a\"}") + " {\"mask\"}"
        with open(f"{inference_path}/prompt.txt", "w") as f:
            f.write(prompt)

        model = self.base_model.split('-')[0]

        # inference
        if self.hyperparams.cluster:
            os.chdir(f"{self.method_path}")
            os.system(f"python emb_prompt_cluster.py --dataset {dataset_name} --model {model} --model_name_or_path {self.base_model} "
                      f"--select {self.hyperparams.select} --seed {self.hyperparams.random_state} --test_mode")
        else:
            os.chdir(f"{self.method_path}")
            os.system(f"python emb_prompt.py --dataset {dataset_name} --model {model} --model_name_or_path {self.base_model} "
                      f"--select {self.hyperparams.select} --seed {self.hyperparams.random_state}")

    def load_pred(self, dataset_name):
        output_dir = os.path.join(self.method_path, "inference", f"{dataset_name}")
        with open(os.path.join(output_dir, "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
        return pred_labels

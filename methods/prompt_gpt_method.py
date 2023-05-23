import os
import json
from dataclasses import dataclass
from utils import Dataset, Labels
from dataclasses_json import dataclass_json
from typing import Optional

# support GPT-2 model
# If you want to run bert-based prompting please refer to prompt_method.py

# if you choose to customize your own prompt, please ensure that it contains
# only a single {} bracket for the input text. We will then add the possible
# label names after your prompt to generate the prediction, in order to
# ensure consistency with bert-based approaches.

# if you customize your own prompt and switch on --dcpmi,
# it's better to also provide the unconditional prompt,
# or we would use a simple way to get it through cutting the (default) prompt by '\n'.
# e.g., the default prompt is " text: {}\n label:"
# then the default unconditional prompt is "\n label:"

@dataclass_json
@dataclass
class prompt_gptHyperparams:
    protocal: bool = False # use prototypical calibration or not
    max_iter: int = 100 # How many times the GMM is repeated, only for prototypical calibration
    dcpmi: bool = False # use dcpmi or not
    prompt: Optional[str] = None # you can overwrite the prompt here
    uncond_prompt: Optional[str] = None # unconditional prompt
    batch: int = 10 # batch_size, not important
    random_state: int = 42

class prompt_gpt():
    def __init__(self, hyperparams: prompt_gptHyperparams, base_model):
        retval = os.getcwd()
        self.method_path = os.path.join(retval, "external", "prompt_gpt")
        self.hyperparams = hyperparams
        self.base_model = base_model

    def apply(self, dataset_name, train_dataset: Dataset, train_label: Labels):
        # apply() only for prototypical calibration

        if self.hyperparams.protocal:
            assert train_dataset.prompt is not None or self.hyperparams.prompt is not None
            dataset_path = os.path.join(self.method_path, "datasets", dataset_name)
            os.system(f"mkdir -p {dataset_path}")
            with open(f"{dataset_path}/classes.txt", "w") as f:
                for line in train_dataset.label_names:
                    f.write(line)
                    f.write("\n")
            with open(f"{dataset_path}/train_dataset.txt", "w") as f:
                for line in train_dataset.texts:
                    f.write(line)
                    f.write("\n")
            with open(f"{dataset_path}/prompt.txt", "w") as f:
                f.write(self.hyperparams.prompt if self.hyperparams.prompt is not None else train_dataset.prompt)
            if self.hyperparams.uncond_prompt is not None:
                with open(f"{dataset_path}/uncond_prompt.txt", "w") as f:
                    f.write(self.hyperparams.uncond_prompt)

            # train
            os.chdir(f"{self.method_path}")
            os.system(f"python ProtoCal.py {dataset_name} --model {self.base_model} --split train "
                      f"--seed {self.hyperparams.random_state} --batch {self.hyperparams.batch} "
                      f"--max_iter {self.hyperparams.max_iter} "
                      f"{'' if not self.hyperparams.dcpmi else '--dcpmi'} "
                      f"{'' if self.hyperparams.uncond_prompt is None else '--uncond_prompt_dir ' + dataset_path + '/uncond_prompt.txt'}")

    def inference(self, dataset_name, test_dataset: Dataset):
        inference_path = os.path.join(self.method_path, "datasets", dataset_name + "_test")
        os.system(f"mkdir -p {inference_path}")
        assert test_dataset.prompt is not None or self.hyperparams.prompt is not None
        if self.hyperparams.protocal:
            with open(f"{inference_path}/classes.txt", "w") as f:
                for line in test_dataset.label_names:
                    f.write(line)
                    f.write("\n")
            with open(f"{inference_path}/test_dataset.txt", "w") as f:
                for line in test_dataset.texts:
                    f.write(line)
                    f.write("\n")
            with open(f"{inference_path}/prompt.txt", "w") as f:
                f.write(self.hyperparams.prompt if self.hyperparams.prompt is not None else test_dataset.prompt)
            if self.hyperparams.uncond_prompt is not None:
                with open(f"{inference_path}/uncond_prompt.txt", "w") as f:
                    f.write(self.hyperparams.uncond_prompt)

            # inference
            os.chdir(f"{self.method_path}")
            os.system(f"python ProtoCal.py {dataset_name}_test --model {self.base_model} --split test "
                      f"--seed {self.hyperparams.random_state} --batch {self.hyperparams.batch} "
                      f"--max_iter {self.hyperparams.max_iter} --test_mode "
                      f"{'' if not self.hyperparams.dcpmi else '--dcpmi'} "
                      f"{'' if self.hyperparams.uncond_prompt is None else '--uncond_prompt_dir ' + inference_path + '/uncond_prompt.txt'}")
        else:
            with open(f"{inference_path}/classes.txt", "w") as f:
                for line in test_dataset.label_names:
                    f.write(line)
                    f.write("\n")
            with open(f"{inference_path}/test_dataset.txt", "w") as f:
                for line in test_dataset.texts:
                    f.write(line)
                    f.write("\n")
            with open(f"{inference_path}/prompt.txt", "w") as f:
                f.write(self.hyperparams.prompt if self.hyperparams.prompt is not None else test_dataset.prompt)
            if self.hyperparams.uncond_prompt is not None:
                with open(f"{inference_path}/uncond_prompt.txt", "w") as f:
                    f.write(self.hyperparams.uncond_prompt)

            # inference
            os.chdir(f"{self.method_path}")
            os.system(f"python score.py {dataset_name}_test --model {self.base_model} --split test "
                      f"--seed {self.hyperparams.random_state} --batch {self.hyperparams.batch} "
                      f"{'' if not self.hyperparams.dcpmi else '--dcpmi'} "
                      f"{'' if self.hyperparams.uncond_prompt is None else '--uncond_prompt_dir ' + inference_path + '/uncond_prompt.txt'}")

    def load_pred(self, dataset_name):
        output_dir = os.path.join(self.method_path, "inference", f"{dataset_name}_test")
        with open(os.path.join(output_dir, "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
        return pred_labels

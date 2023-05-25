import os
import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class xclassHyperparams:
    vocab_min_occurrence: int = 1 # only consider words that appear at least vocab_min_occurrence times
    layer: int = -1 # last layer
    T: int = 100 # number of keywords for each class
    attention_mechanism: str = "mixture" # attention mechanism in xclass
    pca: int = 64 # number of dimensions projected to in PCA, -1 means not doing PCA
    cluster_method: str = "gmm" # choice=["gmm", "kmeans", "none"]
    confidence_threshold: float = 0.5 # confidence threshold for pseudo labels
    random_state: int = 42

class xclass():
    def __init__(self, hyperparams: xclassHyperparams, base_model):
        retval = os.getcwd()
        self.method_path = os.path.join(retval, "external", "xclass")
        self.hyperparams = hyperparams
        self.base_model = base_model

    def apply(self, dataset_name, train_dataset, train_label):
        # not support prompt
        xclass_dataset_path = os.path.join(self.method_path, "data", "datasets", dataset_name)
        os.system(f"mkdir -p {xclass_dataset_path}")
        with open(f"{xclass_dataset_path}/classes.txt", "w") as f:
            for line in train_dataset.label_names:
                f.write(line)
                f.write("\n")
        with open(f"{xclass_dataset_path}/dataset.txt", "w") as f:
            for line in train_dataset.texts:
                f.write(line)
                f.write("\n")

        if train_label.labels is not None:
            with open(f"{xclass_dataset_path}/labels.txt", "w") as f:
                for line in train_label.labels:
                    f.write(str(line))
                    f.write("\n")

        # train
        os.chdir(f"{self.method_path}/scripts")
        os.system(f"python static_representations.py --dataset_name {dataset_name} --lm_type {self.base_model} "
                  f"--vocab_min_occurrence {self.hyperparams.vocab_min_occurrence} --layer {self.hyperparams.layer} "
                  f"--random_state {self.hyperparams.random_state}")
        os.system(f"python class_oriented_document_representations.py --dataset_name {dataset_name} --lm_type {self.base_model} "
                  f"--attention_mechanism {self.hyperparams.attention_mechanism} --layer {self.hyperparams.layer} "
                  f"--T {self.hyperparams.T} --random_state {self.hyperparams.random_state}")
        os.system(f"python document_class_alignment.py --dataset_name {dataset_name} --pca {self.hyperparams.pca} "
                  f"--lm_type {self.base_model}-{self.hyperparams.layer} --cluster_method {self.hyperparams.cluster_method} "
                  f"--document_repr_type {self.hyperparams.attention_mechanism}-{self.hyperparams.T} "
                  f"--random_state {self.hyperparams.random_state}")
        os.system(f"python prepare_text_classifer_training.py --dataset_name {dataset_name} "
                  f"--suffix pca{self.hyperparams.pca}.clus{self.hyperparams.cluster_method}.{self.base_model}-{self.hyperparams.layer}.{self.hyperparams.attention_mechanism}-{self.hyperparams.T}.{self.hyperparams.random_state} "
                  f"--confidence_threshold {self.hyperparams.confidence_threshold}")

    def inference(self, dataset_name, test_dataset):
        xclass_inference_path = os.path.join(self.method_path, "data", "datasets", dataset_name + "_test")
        os.system(f"mkdir -p {xclass_inference_path}")
        with open(f"{xclass_inference_path}/classes.txt", "w") as f:
            for line in test_dataset.label_names:
                f.write(line)
                f.write("\n")
        with open(f"{xclass_inference_path}/dataset.txt", "w") as f:
            for line in test_dataset.texts:
                f.write(line)
                f.write("\n")

        # inference
        os.system(f"sh inference.sh {dataset_name} "
                  f"pca{self.hyperparams.pca}.clus{self.hyperparams.cluster_method}.{self.base_model}-{self.hyperparams.layer}.{self.hyperparams.attention_mechanism}-{self.hyperparams.T}.{self.hyperparams.random_state}.{self.hyperparams.confidence_threshold}")

    def load_pred(self, dataset_name):
        output_dir = os.path.join(self.method_path, "inference", dataset_name)
        with open(os.path.join(output_dir, "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
        return pred_labels

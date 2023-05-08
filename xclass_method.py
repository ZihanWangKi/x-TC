import os
import json

gpu=7

class xclass():
    def __init__(self):
        self.method_path = os.path.join("external", "xclass")

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
        os.system("sh apply.sh {} {}".format(gpu, dataset_name))

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
        os.system("sh inference.sh {} {}".format(gpu, dataset_name))

    def load_pred(self, dataset_name):
        output_dir = os.path.join(self.method_path, "inference", dataset_name)
        with open(os.path.join(output_dir, "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
        return pred_labels

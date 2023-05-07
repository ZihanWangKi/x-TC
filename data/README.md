# Datasets for x-TC

### Format
1. All datasets should be stored in the folder prefixed with its name, and contain three files:
    - `train_text.txt`: the texts for unsupervised training
    - `label_names.txt`: the label names for testing
    - `test_text.txt`: the texts for testing
    - `test_label.txt`: the labels for testing
    - (optional) `train_label.txt`: the labels for unsupervised training, can be used for verification purposes
    - (optional) `prompt.txt`: the textual prompt for prompting, this can also be overwritten in method arguments

2. We provide a sample dataset from AG's News Topic Classification dataset, other datasets can be downloaded: xxx
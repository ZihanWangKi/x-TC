import argparse
import json
import os
import pickle

import numpy as np

from preprocessing_utils import load_classnames, load_labels
from utils import (DATA_FOLDER_PATH, FINETUNE_MODEL_PATH,
                   INTERMEDIATE_DATA_FOLDER_PATH, cosine_similarity_embeddings,
                   evaluate_predictions)


def evaluate(dataset, stage, suffix=None, lm=None):
    data_dir = os.path.join(DATA_FOLDER_PATH, dataset)
    gold_labels = load_labels(data_dir)
    classes = load_classnames(data_dir)
    if stage == "Rep":
        with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset, f"document_repr_lm-{suffix}.pk"), "rb") as f:
            dictionary = pickle.load(f)
            document_representations = dictionary["document_representations"]
            class_representations = dictionary["class_representations"]
            repr_prediction = np.argmax(cosine_similarity_embeddings(document_representations, class_representations),
                                        axis=1)
            evaluate_predictions(gold_labels, repr_prediction)
    elif stage == "Align":
        with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset, f"data.{suffix}.pk"), "rb") as f:
            dictionary = pickle.load(f)
            documents_to_class = dictionary["documents_to_class"]
            evaluate_predictions(gold_labels, documents_to_class)
    else:
        gold_labels = load_labels(data_dir + "_test")
        with open(os.path.join(FINETUNE_MODEL_PATH, lm + "_" + suffix, "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
        se_gold_labels = []
        se_pred_labels = []
        num = [0 for _ in range(100)]
        yes = 0
        for i in range(len(gold_labels)):
            if num[gold_labels[i]] < 4:
                num[gold_labels[i]] += 1
                yes = yes + 1 if gold_labels[i] == pred_labels[i] else yes
                #se_gold_labels.append(gold_labels[i])
                #se_pred_labels.append(pred_labels[i])
        #print(len(gold_labels))
        print(1.0*yes/(4*26))
        #evaluate_predictions(se_gold_labels, se_pred_labels)


if __name__ == '__main__':
    evaluate("NYT-fine", "final", "pca64.clusgmm.bbu-12.mixture-100.42.0.5", "bbu")

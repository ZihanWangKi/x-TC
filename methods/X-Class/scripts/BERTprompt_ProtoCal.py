import argparse
import os
import pickle as pk
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from preprocessing_utils import *
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from utils import *
from transformers import BertTokenizer, BertForMaskedLM

def prepare_sentence(tokenizer, text, prompt):
    # setting for BERT
    model_max_tokens = 500 # for prompt
    has_sos_eos = True
    ######################
    max_tokens = model_max_tokens
    if has_sos_eos:
        max_tokens -= 2

    if not hasattr(prepare_sentence, "sos_id"):
        prepare_sentence.sos_id, prepare_sentence.eos_id = tokenizer.encode("", add_special_tokens=True)
        print(prepare_sentence.sos_id, prepare_sentence.eos_id)

    import copy
    backup = copy.deepcopy(text)
    r = prompt.find('}')
    left_prompt = prompt[:r+1]
    right_prompt = prompt[r+1: ]
    text = left_prompt.format(text)

    ids = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    ids = [prepare_sentence.sos_id] + ids + tokenizer.encode(right_prompt) + \
          [tokenizer._convert_token_to_id(tokenizer.mask_token), prepare_sentence.eos_id]

    return len(ids) - 2, torch.tensor([ids]).long()


def main(args):
    dataset = load(args.dataset_name)
    print("Finish reading data")

    data_folder = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name)
    if args.lm_type == 'bbu' or args.lm_type == 'blu':
        dataset["class_names"] = [x.lower() for x in dataset["class_names"]]

    os.makedirs(data_folder, exist_ok=True)
    with open(os.path.join(data_folder, "dataset.pk"), "wb") as f:
        pk.dump(dataset, f)

    data = dataset["cleaned_text"]
    if args.lm_type == 'bbu' or args.lm_type == 'blu':
        data = [x.lower() for x in data]
    model_class, tokenizer_class, pretrained_weights = MODELS[args.lm_type]
    model_class = BertForMaskedLM

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    model.eval()
    model.cuda()

    with open(os.path.join(DATA_FOLDER_PATH, args.dataset_name, 'prompt.txt'), "r") as fp:
        prompt = fp.read()

    vecs = []
    for text in tqdm(data):
        masked_index, tokens_tensor = prepare_sentence(tokenizer, text, prompt)
        with torch.no_grad():
            with torch.no_grad():
                outputs = model(tokens_tensor.cuda())
        predictions = outputs[0]
        vec = []
        for i in range(len(dataset["class_names"])):
            cls_name = dataset["class_names"][i]
            val = predictions[0, masked_index, tokenizer._convert_token_to_id(cls_name)].item()
            vec.append(val)
        #_, pred_cls, _ = sorted(Q)[0]
        #pred.append(pred_cls)
        vecs.append(vec)

    vecs = np.array(vecs)
    max_cla = -1000000
    best_seed = 0
    for seed in range(args.iter):
        gmm = GaussianMixture(n_components=len(dataset["class_names"]), random_state=seed)
        gmm.fit(vecs)
        documents_to_class = gmm.predict(vecs)
        centers = gmm.means_
        row_ind, col_ind = linear_sum_assignment(centers.max() - centers)
        cla = centers[row_ind, col_ind].sum()
        if cla > max_cla:
            max_cla = cla
            best_seed = seed

    gmm = GaussianMixture(n_components=len(dataset["class_names"]), random_state=best_seed)
    gmm.fit(vecs)
    documents_to_class = gmm.predict(vecs)
    centers = gmm.means_
    row_ind, col_ind = linear_sum_assignment(centers.max() - centers)
    print("best seed : " + str(best_seed))
    print("class center :")
    print(centers)

    test_dataset_name = args.dataset_name+"_test"
    data_dir = os.path.join(DATA_FOLDER_PATH, test_dataset_name)
    text = load_text(data_dir)
    text = [s.strip() for s in text]
    text_statistics(text, "raw_txt")

    cleaned_text = [clean_str(doc) for doc in text]
    #print(f"Cleaned {len(clean_html.clean_links)} html links")
    text_statistics(cleaned_text, "cleaned_txt")

    data = cleaned_text
    if args.lm_type == 'bbu' or args.lm_type == 'blu':
        data = [x.lower() for x in data]

    vecs = []
    for text in tqdm(data):
        masked_index, tokens_tensor = prepare_sentence(tokenizer, text, prompt)
        with torch.no_grad():
            with torch.no_grad():
                outputs = model(tokens_tensor.cuda())
        predictions = outputs[0]
        vec = []
        for i in range(len(dataset["class_names"])):
            cls_name = dataset["class_names"][i]
            val = predictions[0, masked_index, tokenizer._convert_token_to_id(cls_name)].item()
            vec.append(val)
        vec = np.array(vec)
        vec = np.exp(vec)
        vec = np.log(vec / vec.sum())
        vecs.append(vec)

    vecs = np.array(vecs)
    documents_to_class = gmm.predict(vecs)
    pred = [col_ind[documents_to_class[i]] for i in range(len(vecs))]

    data_dir = os.path.join(DATA_FOLDER_PATH, args.dataset_name+"_test")
    gold_labels = load_labels(data_dir)
    evaluate_predictions(gold_labels, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, default='bbu')
    parser.add_argument("--vocab_min_occurrence", type=int, default=5)
    # last layer of BERT
    parser.add_argument("--iter", type=int, default=100)
    args = parser.parse_args()
    print(vars(args))
    main(args)

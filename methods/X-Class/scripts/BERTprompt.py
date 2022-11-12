import argparse
import os
import pickle as pk
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from preprocessing_utils import load, load_labels
from utils import INTERMEDIATE_DATA_FOLDER_PATH, DATA_FOLDER_PATH, MODELS, tensor_to_numpy, evaluate_predictions
from transformers import BertTokenizer, BertForMaskedLM

def prepare_sentence(tokenizer, text, prompt):
    # setting for BERT
    model_max_tokens = 512 # for prompt
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
    #left_prompt = prompt[:r+1]
    #right_prompt = prompt[r+1: ]
    text = prompt.format(text) + "[MASK]"
    if text > 500: print(text)

    ids = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    #ids = [prepare_sentence.sos_id] + ids + tokenizer.encode(right_prompt) + \
    #      [tokenizer._convert_token_to_id(tokenizer.mask_token), prepare_sentence.eos_id]

    return torch.tensor([ids]).long()


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

    #######################
    # masked LM check
    mask = tokenizer.mask_token
    text = 'United [MASK] is a country'
    ids = tokenizer.encode(text)
    print(ids)
    ids = torch.tensor([ids]).long().cuda()
    with torch.no_grad():
        output = model(ids.cuda())
    predictions = output[0]
    masked_index = (ids == tokenizer.mask_token_id).nonzero()[0, 1]
    _, predicted_index = torch.topk(predictions[0, masked_index], k=5)
    predicted_token = [tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in predicted_index]
    print(predicted_token)
    #######################

    with open(os.path.join(DATA_FOLDER_PATH, args.dataset_name, 'prompt.txt'), "r") as fp:
        prompt = fp.read()

    pred = []
    for text in tqdm(data):
        tokens_tensor = prepare_sentence(tokenizer, text, prompt)
        masked_index = (ids == tokenizer.mask_token_id).nonzero()[0, 1]
        with torch.no_grad():
            output = model(tokens_tensor.cuda())
        predictions = output[0]
        Q = []
        for i in range(len(dataset["class_names"])):
            cls_name = dataset["class_names"][i]
            val = predictions[0, masked_index, tokenizer._convert_token_to_id(cls_name)].item()
            Q.append((-val, i, cls_name))
        _, pred_cls, _ = sorted(Q)[0]
        pred.append(pred_cls)

    data_dir = os.path.join(DATA_FOLDER_PATH, args.dataset_name)
    gold_labels = load_labels(data_dir)
    evaluate_predictions(gold_labels, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, default='bbu')
    parser.add_argument("--vocab_min_occurrence", type=int, default=5)
    # last layer of BERT
    parser.add_argument("--layer", type=int, default=12)
    args = parser.parse_args()
    print(vars(args))
    main(args)

import argparse
import os
import pickle as pk
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from preprocessing_utils import load, load_labels
from utils import INTERMEDIATE_DATA_FOLDER_PATH, DATA_FOLDER_PATH, MODELS, tensor_to_numpy, evaluate_predictions
from transformers import ElectraTokenizer, ElectraForPreTraining, BertTokenizer, BertForMaskedLM, RobertaForMaskedLM, RobertaTokenizer, BartTokenizer, BartForConditionalGeneration

def prepare_sentence(args, tokenizer, text, prompt):
    # setting for BERT
    if args.lm_type == "bart-base" or args.lm_type == "bart-large":
        model_max_tokens = 1024
    else:
        model_max_tokens = 512

    import copy
    backup = copy.deepcopy(text)

    l = 1
    r= len(text)
    while l <= r:
        mid = (l+r) >> 1
        text = backup[:mid]
        text = prompt.format(text)
        if args.add_mask:
            text = text.strip()
            if args.lm_type == "roberta-large" or args.lm_type == "roberta-base" or args.lm_type == "bart-base"  or args.lm_type == "bart-large":
                text += ' <mask>'
            else:
                text += ' [MASK]'
        ids = tokenizer.encode(text, truncation=True, max_length=model_max_tokens)
        if tokenizer.mask_token_id in ids:
            good_ids = ids
            l = mid + 1
        else:
            r = mid - 1

    #ids = [prepare_sentence.sos_id] + ids + tokenizer.encode(right_prompt) + \
    #      [tokenizer._convert_token_to_id(tokenizer.mask_token), prepare_sentence.eos_id]

    return torch.tensor([good_ids]).long()


def main(args):
    dataset = load(args.dataset_name)
    print("Finish reading data")

    data_folder = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name)
    dataset["class_names"] = [x for x in dataset["class_names"]]

    os.makedirs(data_folder, exist_ok=True)
    with open(os.path.join(data_folder, "dataset.pk"), "wb") as f:
        pk.dump(dataset, f)

    data = dataset["cleaned_text"]
    if args.lm_type == 'bbu' or args.lm_type == 'blu':
        data = [x.lower() for x in data]

    if args.lm_type == "roberta-base" or args.lm_type == "roberta-large":
        tokenizer_class = RobertaTokenizer
        pretrained_weights = args.lm_type
        model_class = RobertaForMaskedLM
    elif args.lm_type == "bart-base" or args.lm_type == "bart-large":
        tokenizer_class = BartTokenizer
        pretrained_weights = "facebook/" + args.lm_type
        model_class = BartForConditionalGeneration
    elif args.lm_type == "electra-base" or args.lm_type == "electra-small" or args.lm_type == "electra-large":
        tokenizer_class = ElectraTokenizer
        pretrained_weights = "google/" + args.lm_type + "-discriminator"
        model_class = ElectraForPreTraining
    else:
        model_class, tokenizer_class, pretrained_weights = MODELS[args.lm_type]
        model_class = BertForMaskedLM

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    model.eval()
    model.cuda()

    for cls in dataset["class_names"]:
        print(tokenizer.encode(cls))

    #######################
    # check
    if args.lm_type != "electra-base" and args.lm_type != "electra-small" and args.lm_type != "electra-large":
        print("MLM check...")
        mask = tokenizer.mask_token
        text = 'United [MASK] is a country. New York is a city.'
        if args.lm_type == "roberta-large" or args.lm_type == "roberta-base" or args.lm_type == "bart-base" or args.lm_type == "bart-large":
            text = 'United <mask> is a country. New York is a city.'
        print(text)
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
    else:
        print("electra check...")
        text1 = 'I like to eat apple.'
        text2 = 'I like to drink apple.'
        print(text1)
        ids = tokenizer.encode(text1)
        ids = torch.tensor([ids]).long().cuda()
        with torch.no_grad():
            output = model(ids.cuda())
        predictions = output[0]
        print(predictions)
        print(text2)
        ids = tokenizer.encode(text2)
        ids = torch.tensor([ids]).long().cuda()
        with torch.no_grad():
            output = model(ids.cuda())
        predictions = output[0]
        print(predictions)
    #######################

    with open(os.path.join(DATA_FOLDER_PATH, args.dataset_name, 'prompt.txt'), "r") as fp:
        prompt = fp.read()

    pred = []
    if args.lm_type != "electra-base" and args.lm_type != "electra-small" and args.lm_type != "electra-large":
        for text in tqdm(data):
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt)
            masked_index = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
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
    else:
        for text in tqdm(data):
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt)
            masked_index = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
            Q = []
            for i in range(len(dataset["class_names"])):
                cls_name = dataset["class_names"][i]
                tokens_tensor[0, masked_index] = tokenizer._convert_token_to_id(cls_name)
                with torch.no_grad():
                    output = model(tokens_tensor.cuda())
                predictions = output[0]
                val = predictions[masked_index].item()
                Q.append((val, i, cls_name))
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
    parser.add_argument("--add_mask", action='store_true', default=False)
    args = parser.parse_args()
    print(vars(args))
    main(args)

import argparse
import json
import os
import pickle as pk
from collections import defaultdict
import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from preprocessing_utils import load, load_labels
from utils import DATA_FOLDER_PATH, INFERENCE_PATH
from transformers import BertTokenizer, BertForMaskedLM, RobertaForMaskedLM,\
    RobertaTokenizer, BartTokenizer, BartForConditionalGeneration

lm_types = ['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased',
            'roberta-base', 'roberta-large', 'bart-base', 'bart-large']

def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.random_state)
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.cuda.manual_seed_all(args.random_state)

def prepare_sentence(args, tokenizer, text, prompt, uncond=False):
    # setting for lm
    if args.lm_type.startswith("bart"):
        model_max_tokens = 1024
    else:
        model_max_tokens = 512

    has_sos_eos = True
    max_tokens = model_max_tokens
    if has_sos_eos:
        max_tokens -= 2
    if not hasattr(prepare_sentence, "sos_id"):
        prepare_sentence.sos_id, prepare_sentence.eos_id = tokenizer.encode("", add_special_tokens=True)

    prompt_ = copy.deepcopy(prompt)
    if uncond:
        if args.uncond_prompt_dir is None:
            prompt_ =  '\n' + prompt.split('\n')[-1]
        else:
            with open(args.uncond_prompt_dir, "r") as fp:
                prompt_ = fp.read()

    input = prompt_.format(text)
    if args.lm_type.startswith("roberta") or args.lm_type.startswith("bart"):
        input += ' <mask>'
    else:
        input += ' [MASK]'
    input_tokens = [prepare_sentence.sos_id] + tokenizer.tokenize(input)[-max_tokens:] + [prepare_sentence.eos_id]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    return torch.tensor([input_ids]).long()


def main(args):
    set_seed(args)
    dataset = load(args.dataset_name)
    print("Finish reading data")

    dataset["class_names"] = [x for x in dataset["class_names"]]

    data = dataset["cleaned_text"]
    if args.lm_type == 'bert-base-uncased' or args.lm_type == 'bert-large-uncased':
        data = [x.lower() for x in data]
        dataset["class_names"] = [x.lower() for x in dataset["class_names"]]

    if args.lm_type.startswith("roberta"):
        tokenizer_class = RobertaTokenizer
        pretrained_weights = args.lm_type
        model_class = RobertaForMaskedLM
    elif args.lm_type.startswith("bart"):
        tokenizer_class = BartTokenizer
        pretrained_weights = "facebook/" + args.lm_type
        model_class = BartForConditionalGeneration
    else:
        tokenizer_class = BertTokenizer
        pretrained_weights = args.lm_type
        model_class = BertForMaskedLM

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    model.eval()
    model.cuda()

    for cls in dataset["class_names"]:
        tmp = tokenizer.tokenize(cls)
        assert len(tmp) == 1, f"class name {cls} is more than one token in {args.lm_type}."

    with open(os.path.join(DATA_FOLDER_PATH, args.dataset_name, 'prompt.txt'), "r") as fp:
        prompt = fp.read()

    pred = []
    if not args.dcpmi:
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
            masked_index_1 = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
            with torch.no_grad():
                output_1 = model(tokens_tensor.cuda())
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt, uncond=True)
            masked_index_2 = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
            with torch.no_grad():
                output_2 = model(tokens_tensor.cuda())
            predictions_1 = output_1[0]
            predictions_2 = output_2[0]
            Q = []
            for i in range(len(dataset["class_names"])):
                cls_name = dataset["class_names"][i]
                val = predictions_1[0, masked_index_1, tokenizer._convert_token_to_id(cls_name)].item() \
                      - predictions_2[0, masked_index_2, tokenizer._convert_token_to_id(cls_name)].item()
                Q.append((-val, i, cls_name))
            _, pred_cls, _ = sorted(Q)[0]
            pred.append(pred_cls)

    inference_path = os.path.join(INFERENCE_PATH, args.dataset_name)
    os.system(f"mkdir -p {inference_path}")
    json.dump(pred, open(f"{inference_path}/eval_labels.json", "w"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, choices=lm_types)
    parser.add_argument("--dcpmi", action='store_true', default=False)
    parser.add_argument("--uncond_prompt_dir", type=str, default=None)
    # if you customize your own prompt and switch on --dcpmi, it's better to also input the unconditional prompt,
    # or we would use a simple way to get it through cutting the (default) prompt by '\n'.
    # e.g., the default prompt:
    # text: {}\n label:
    # then the default unconditional prompt is:
    #\n label:
    args = parser.parse_args()
    print(vars(args))
    main(args)

import argparse
import copy
import json
import os
import pickle
import pickle as pk
import random
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from preprocessing_utils import load, load_labels
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


def train(args):
    set_seed(args)
    dataset = load(args.exp_name, args.dataset_name)
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

    DATA_FOLDER_PATH = os.path.join(args.exp_name, "datasets")
    with open(os.path.join(DATA_FOLDER_PATH, args.dataset_name, 'prompt.txt'), "r") as fp:
        prompt = fp.read()

    vecs = []
    if not args.dcpmi:
        for text in tqdm(data):
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt)
            masked_index = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
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
            vec = []
            for i in range(len(dataset["class_names"])):
                cls_name = dataset["class_names"][i]
                val = predictions_1[0, masked_index_1, tokenizer._convert_token_to_id(cls_name)].item() \
                      - predictions_2[0, masked_index_2, tokenizer._convert_token_to_id(cls_name)].item()
                vec.append(val)
            vec = np.array(vec)
            vec = np.exp(vec)
            vec = np.log(vec / vec.sum())
            vecs.append(vec)

    vecs = np.array(vecs)
    max_cla = -1000000
    best_seed = 0
    for seed in range(args.max_iter):
        gmm = GaussianMixture(n_components=len(dataset["class_names"]), random_state=seed)
        gmm.fit(vecs)
        documents_to_class = gmm.predict(vecs)
        centers = gmm.means_
        row_ind, col_ind = linear_sum_assignment(centers.max() - centers)
        cla = centers[row_ind, col_ind].sum()
        # print(cla, centers)
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

    MODEL_PATH = os.path.join(args.exp_name, "models")
    os.system(f"mkdir -p {MODEL_PATH}")
    with open(f'{MODEL_PATH}/gmm_{args.dataset_name}_test_{args.lm_type}.pkl', 'wb') as f:
        pickle.dump(gmm, f)



def test(args):
    MODEL_PATH = os.path.join(args.exp_name, "models")
    with open(f'{MODEL_PATH}/gmm_{args.dataset_name}_{args.lm_type}.pkl', 'rb') as f:
        gmm = pickle.load(f)

    centers = gmm.means_
    row_ind, col_ind = linear_sum_assignment(centers.max() - centers)

    dataset = load(args.exp_name, args.dataset_name)
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

    DATA_FOLDER_PATH = os.path.join(args.exp_name, "datasets")
    with open(os.path.join(DATA_FOLDER_PATH, args.dataset_name, 'prompt.txt'), "r") as fp:
        prompt = fp.read()

    vecs = []
    if not args.dcpmi:
        for text in tqdm(data):
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt)
            masked_index = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
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
            vec = []
            for i in range(len(dataset["class_names"])):
                cls_name = dataset["class_names"][i]
                val = predictions_1[0, masked_index_1, tokenizer._convert_token_to_id(cls_name)].item() \
                      - predictions_2[0, masked_index_2, tokenizer._convert_token_to_id(cls_name)].item()
                vec.append(val)
            vec = np.array(vec)
            vec = np.exp(vec)
            vec = np.log(vec / vec.sum())
            vecs.append(vec)

    vecs = np.array(vecs)
    documents_to_class = gmm.predict(vecs)
    pred = [int(col_ind[documents_to_class[i]]) for i in range(len(vecs))]

    inference_path = os.path.join(args.exp_name, "inference", args.dataset_name)
    os.system(f"mkdir -p {inference_path}")
    json.dump(pred, open(f"{inference_path}/eval_labels.json", "w"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, choices=lm_types)
    parser.add_argument("--dcpmi", action='store_true', default=False)
    parser.add_argument("--uncond_prompt_dir", type=str, default=None)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--test_mode", action='store_true', default=False)
    parser.add_argument("--exp_name", type=str, required=True)

    args = parser.parse_args()
    print(vars(args))

    if not args.test_mode:
        train(args)
    else:
        test(args)

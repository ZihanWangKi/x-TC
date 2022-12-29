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
    dataset["class_names"] = [x.lower() for x in dataset["class_names"]]

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

    with open(os.path.join(DATA_FOLDER_PATH, args.dataset_name, 'prompt.txt'), "r") as fp:
        prompt = fp.read()

    vecs = []
    pred = []
    if args.lm_type != "electra-base" and args.lm_type != "electra-small" and args.lm_type != "electra-large":
        for text in tqdm(data):
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt)
            masked_index = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
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
            # _, pred_cls, _ = sorted(Q)[0]
            pred.append(np.argmax(vec))
            vecs.append(vec)
    else:
        for text in tqdm(data):
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt)
            masked_index = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
            vec = []
            for i in range(len(dataset["class_names"])):
                cls_name = dataset["class_names"][i]
                tokens_tensor[0, masked_index] = tokenizer._convert_token_to_id(cls_name)
                with torch.no_grad():
                    output = model(tokens_tensor.cuda())
                predictions = output[0]
                val = predictions[masked_index].item()
                vec.append(val)
            vec = np.array(vec)
            vec = np.exp(vec)
            vec = np.log(vec / vec.sum())
            vecs.append(vec)

    if args.lm_type != "electra-base" and args.lm_type != "electra-small" and args.lm_type != "electra-large":
        vecs = np.array(vecs)
        max_cla = -1000000
        best_seed = 0
        if args.magic:
            assignment_matrix = np.zeros((len(pred), len(dataset["class_names"])))
            for i in range(len(pred)):
                assignment_matrix[i][pred[i]] = 1.0

            gmm = GaussianMixture(n_components=len(dataset["class_names"]), warm_start=True)
            gmm.converged_ = "HACK"

            gmm._initialize(vecs, assignment_matrix)
            gmm.lower_bound_ = -np.infty

            gmm.fit(vecs)
            documents_to_class = gmm.predict(vecs)
            centers = gmm.means_
            row_ind, col_ind = linear_sum_assignment(centers.max() - centers)
            print("best seed : " + str(best_seed))
            print("class center :")
            print(centers)
        else:
            for seed in range(args.iter):
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
    else:
        vecs = np.array(vecs)
        min_cla = 1000000
        best_seed = 0
        for seed in range(args.iter):
            gmm = GaussianMixture(n_components=len(dataset["class_names"]), random_state=seed)
            gmm.fit(vecs)
            documents_to_class = gmm.predict(vecs)
            centers = gmm.means_
            row_ind, col_ind = linear_sum_assignment(centers)
            cla = centers[row_ind, col_ind].sum()
            if cla < min_cla:
                min_cla = cla
                best_seed = seed

        gmm = GaussianMixture(n_components=len(dataset["class_names"]), random_state=best_seed)
        gmm.fit(vecs)
        documents_to_class = gmm.predict(vecs)
        centers = gmm.means_
        row_ind, col_ind = linear_sum_assignment(centers)
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
    if args.lm_type != "electra-base" and args.lm_type != "electra-small" and args.lm_type != "electra-large":
        for text in tqdm(data):
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt)
            masked_index = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]

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
    else:
        for text in tqdm(data):
            tokens_tensor = prepare_sentence(args, tokenizer, text, prompt)
            masked_index = (tokens_tensor == tokenizer.mask_token_id).nonzero()[0, 1]
            vec = []
            for i in range(len(dataset["class_names"])):
                cls_name = dataset["class_names"][i]
                tokens_tensor[0, masked_index] = tokenizer._convert_token_to_id(cls_name)
                with torch.no_grad():
                    output = model(tokens_tensor.cuda())
                predictions = output[0]
                val = predictions[masked_index].item()
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
    parser.add_argument("--add_mask", action='store_true', default=False)
    parser.add_argument("--magic", action='store_true', default=False)
    args = parser.parse_args()
    os.environ['PYTHONHASHSEED'] = str(args.random_state)
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.cuda.manual_seed_all(args.random_state)
    print(vars(args))
    main(args)

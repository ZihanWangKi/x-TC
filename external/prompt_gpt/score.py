import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import score
import argparse
import random
import numpy as np
import torch
import os

def get_model(model_name):
    if model_name.lower() in ['gpt2', 'gpt2-s', 'gpt2-small', 'gs', 's', 'small']:
        # GPT-2 Small
        model   = GPT2LMHeadModel.from_pretrained('gpt2').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2')
        name    = 'G-S'
    elif model_name.lower() in ['gpt2-m', 'gpt2-medium', 'gm', 'm', 'medium']:
        # GPT-2 Medium
        model   = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-medium')
        name    = 'G-M'
    elif model_name.lower() in ['gpt2-l', 'gpt2-large', 'gl', 'l', 'large']:
        # GPT-2 Large
        model   = GPT2LMHeadModel.from_pretrained('gpt2-large').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-large')
        name    = 'G-L'
    elif model_name.lower() in ['gpt2-xl', 'gxl', 'xl', 'extra-large']:
        # GPT-2 XL
        model   = GPT2LMHeadModel.from_pretrained('gpt2-xl').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-xl')
        name    = 'G-XL'
    else:
        raise ValueError(f'No model {model_name}')
    return model, encoder, name

def load_examples(stem, split, uncond_prompt_dir):
    examples = []

    path = f"{stem}classes.txt"
    with open(path, "r") as fp:
        class_names = list(map(lambda x: x.strip(), fp.readlines()))

    path = f"{stem}prompt.txt"
    with open(path, "r") as fp:
        prompt = fp.read()

    if uncond_prompt_dir is not None:
        path = uncond_prompt_dir
        with open(path, "r") as fp:
            uncond_prompt = fp.read()
    else:
        uncond_prompt = '\n' + prompt.split('\n')[-1]

    path = f"{stem}{split}_dataset.txt"
    with open(path, "r") as fp:
        texts = list(map(lambda x: x.strip(), fp.readlines()))

    #path = f"{stem}{split}_labels.txt"
    #with open(path, "r") as fp:
    #    text_labels = list(map(lambda x: x.strip(), fp.readlines()))

    for i in range(len(texts)):
        #label = int(text_labels[i])
        premise = prompt.format(texts[i])
        uncond_premise = uncond_prompt.format(texts[i])
        options = []
        for h in class_names:
            o = {}
            o['premise'] = premise
            o['hypothesis'] = ' ' + h
            o['uncond_premise'] = uncond_premise
            o['uncond_hypothesis'] = ' ' + h
            options.append(o)
        label = 0 # give fake labels here just to maintain the original framework
        examples.append({'options': options, 'label': label})
    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str, default='gpt2-xl')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dcpmi', action="store_true", default=False)
    parser.add_argument('--uncond_prompt_dir', type=str, default=None)
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    print(args)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, encoder, name = get_model(args.model)
    retval = args.exp_name
    stem = f'{retval}/datasets/{args.dataset}/'

    examples = load_examples(stem, args.split, args.uncond_prompt_dir)

    preds_dict = score(model, args.model, encoder, examples, stem, args.split, args.batch)

    inference_path = os.path.join(retval, 'inference', args.dataset)
    os.system(f"mkdir -p {inference_path}")

    if args.dcpmi:
        preds = preds_dict["dcpmi"]
    else:
        preds = preds_dict["lm"]
    json.dump(preds, open(f"{inference_path}/eval_labels.json", "w"))

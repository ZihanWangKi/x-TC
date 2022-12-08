def get_model(model_name, key_file):
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
    elif model_name.lower() == 'ada' or \
         model_name.lower() == 'babbage' or \
         model_name.lower() == 'curie' or \
         model_name.lower() == 'davinci':
        # GPT-3
        model = name = model_name
        encoder = None
        import openai
        with open(key_file) as f:
            api_key = f.read().strip()
        openai.api_key = api_key
    else:
        raise ValueError(f'No model {model_name}')
    return model, encoder, name

def get_examples(dataset_name, split, stem, n_shot, variant):
    if dataset_name == 'copa':
        from data_loaders import load_examples_copa
        examples = load_examples_copa(f'{stem}copa-{split}.xml')
        closed_label_space = False
    elif dataset_name == 'copa-rev':
        from data_loaders import load_examples_copa_rev
        examples = load_examples_copa_rev(f'{stem}copa-{split}.xml')
        closed_label_space = False
    elif dataset_name == 'storycloze':
        from data_loaders import load_examples_storycloze
        examples = load_examples_storycloze(f'{stem}{split}.tsv')
        closed_label_space = False
    elif dataset_name == 'hellaswag':
        from data_loaders import load_examples_hellaswag
        examples = load_examples_hellaswag(f'{stem}dev.jsonl')
        closed_label_space = False
    elif dataset_name == 'race-m' or \
         dataset_name == 'race-h':
        from data_loaders import load_examples_race
        version = 'high' if dataset_name == 'race-h' else 'middle'
        examples = load_examples_race(stem, split, version)
        closed_label_space = False
    elif dataset_name == 'arc-easy' or \
         dataset_name == 'arc-challenge':
        from data_loaders import load_examples_arc
        examples = load_examples_arc(f'{stem}{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'obqa':
        from data_loaders import load_examples_obqa
        examples = load_examples_obqa(f'{stem}{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'cqa':
        from data_loaders import load_examples_cqa
        if args.split == 'test':
            raise NotImplementedError("CSQA does not release test answers directly, please do not spam their leaderboard either :)")
        else:
            examples = load_examples_cqa(f'{stem}{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'boolq':
        from data_loaders import load_examples_boolq
        examples = load_examples_boolq(f'{stem}dev.jsonl')
        closed_label_space = True
    elif dataset_name == 'rte':
        from data_loaders import load_examples_rte
        examples = load_examples_rte(f'{stem}dev.jsonl')
        closed_label_space = True
    elif dataset_name == 'cb':
        from data_loaders import load_examples_cb
        examples = load_examples_cb(f'{stem}dev.jsonl')
        closed_label_space = True
    elif dataset_name == 'sst-2':
        from data_loaders import load_examples_sst2, load_examples_sst2_variants
        if n_shot > 0:
            examples = load_examples_sst2(f'{stem}{split}.tsv', f'{stem}/train.tsv', n_shot)
        elif variant is not None:
            examples = load_examples_sst2_variants(f'{stem}{split}.tsv', variant)
        else:
            examples = load_examples_sst2(f'{stem}{split}.tsv')
        closed_label_space = True
    elif dataset_name == 'sst-5':
        from data_loaders import load_examples_sst5
        examples = load_examples_sst5(f'{stem}{split}.tsv')
        closed_label_space = True
    elif dataset_name == 'agn':
        from data_loaders import load_examples_agn
        examples = load_examples_agn(f'{stem}{split}.csv')
        closed_label_space = True
    elif dataset_name == 'trec':
        split = 'train' if split == 'dev' else split
        from data_loaders import load_examples_trec
        examples = load_examples_trec(f'{stem}{split}.txt')
        closed_label_space = True
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    return examples, closed_label_space


if __name__ == '__main__':
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from ProtoCal_utils import score
    import argparse
    import random
    import numpy as np
    import torch
    import os
    import pdb
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import linear_sum_assignment

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str, default='xl')
    parser.add_argument('--n_shot', type=int, default=0)
    parser.add_argument('--variant', type=int, default=None)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch', type=int, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key', type=str, default='api.key')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument("--magic", action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    if args.debug:
        pdb.set_trace()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, encoder, name = get_model(args.model, args.key)
    if args.dataset.endswith('-rev'):
        stem = f'data/{args.dataset[:-4]}/'
    else:
        stem = f'data/{args.dataset}/'

    def load_examples(stem, split):
        examples = []
        path = f"{stem}class_names.txt"
        with open(path, "r") as fp:
            class_names = list(map(lambda x: x.strip(), fp.readlines()))
        path = f"{stem}prompt.txt"
        with open(path, "r") as fp:
            prompt = fp.read()
        path = f"{stem}{split}.txt"
        with open(path, "r") as fp:
            texts = list(map(lambda x: x.strip(), fp.readlines()))
        path = f"{stem}{split}_labels.txt"
        with open(path, "r") as fp:
            text_labels = list(map(lambda x: x.strip(), fp.readlines()))
        for i in range(len(texts)):
            label = int(text_labels[i])
            #premise = " text: {}\n topic:".format(texts[i])
            premise = prompt.format(texts[i])
            uncond_premise = '\n' + premise.split('\n')[-1]
            options = []
            for h in class_names:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = ' ' + h.lower()
                o['uncond_premise'] = uncond_premise #'\n topic:'
                o['uncond_hypothesis'] = ' ' + h.lower()
                options.append(o)
            label = label
            examples.append({'options': options, 'label': label})
        return len(class_names), examples

    def load_examples_n_shot(stem, split, n_shot):
        examples = []
        path = f"{stem}class_names.txt"
        with open(path, "r") as fp:
            class_names = list(map(lambda x: x.strip(), fp.readlines()))
        path = f"{stem}prompt.txt"
        with open(path, "r") as fp:
            prompt = fp.read()
        path = f"{stem}{split}.txt"
        with open(path, "r") as fp:
            texts = list(map(lambda x: x.strip(), fp.readlines()))
        path = f"{stem}{split}_labels.txt"
        with open(path, "r") as fp:
            text_labels = list(map(lambda x: x.strip(), fp.readlines()))

        path = f"{stem}n_shot.txt"
        with open(path, "r") as fp:
            n_shot_texts = list(map(lambda x: x.strip(), fp.readlines()))
        path = f"{stem}n_shot_labels.txt"
        with open(path, "r") as fp:
            n_shot_text_labels = list(map(lambda x: x.strip(), fp.readlines()))

        fewshot_examples = []
        for i in range(len(n_shot_texts)):
            l = n_shot_text_labels[i]
            s = n_shot_texts[i]
            fewshot_prefix = prompt.format(s) + ' ' + class_names[int(l)].lower()
            fewshot_examples.append(fewshot_prefix)

        random.shuffle(fewshot_examples)
        fewshot_prefix = ''
        for ex in fewshot_examples:
            fewshot_prefix = fewshot_prefix + '\n' + ex
        fewshot_prefix = fewshot_prefix + '\n'

        for i in range(len(texts)):
            label = int(text_labels[i])
            #premise = " text: {}\n topic:".format(texts[i])
            premise = fewshot_prefix + prompt.format(texts[i])
            uncond_premise = '\n' + premise.split('\n')[-1]
            options = []
            for h in class_names:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = ' ' + h.lower()
                o['uncond_premise'] = uncond_premise #'\n topic:'
                o['uncond_hypothesis'] = ' ' + h.lower()
                options.append(o)
            label = label
            examples.append({'options': options, 'label': label})
        return len(class_names), examples

    if args.n_shot == 0:
        n_class, train_examples = load_examples(stem, "train")
    else:
        n_class, train_examples = load_examples_n_shot(stem, "train", args.n_shot)

    train_vec = score(model, args.model, encoder, train_examples, stem, "train", args.batch)
    max_cla = -1000000
    best_seed = 0
    pred = []
    for i in range(len(train_vec)):
        pred = np.argmax(train_vec[i])
    for seed in range(args.iter):
        if args.magic:
            assignment_matrix = np.zeros((len(pred), n_class))
            for i in range(len(pred)):
                assignment_matrix[i][pred[i]] = 1.0

            gmm = GaussianMixture(n_components=n_class,
                                  random_state=seed, warm_start=True)
            gmm.converged_ = "HACK"

            gmm._initialize(train_vec, assignment_matrix)
            gmm.lower_bound_ = -np.infty
        else:
            gmm = GaussianMixture(n_components=n_class, random_state=seed)
        gmm.fit(train_vec)
        documents_to_class = gmm.predict(train_vec)
        centers = gmm.means_
        row_ind, col_ind = linear_sum_assignment(centers.max() - centers)
        cla = centers[row_ind, col_ind].sum()
        if cla > max_cla:
            max_cla = cla
            best_seed = seed

    gmm = GaussianMixture(n_components=n_class, random_state=best_seed)
    gmm.fit(train_vec)
    documents_to_class = gmm.predict(train_vec)
    centers = gmm.means_
    row_ind, col_ind = linear_sum_assignment(centers.max() - centers)
    print("best seed : " + str(best_seed))
    print("class center :")
    print(centers)

    if args.n_shot == 0:
        _, examples = load_examples(stem, args.split)
    else:
        _, examples = load_examples_n_shot(stem, args.split, args.n_shot)
    test_vec = score(model, args.model, encoder, examples, stem, args.split, args.batch)

    documents_to_class = gmm.predict(test_vec)
    pred = [col_ind[documents_to_class[i]] for i in range(len(test_vec))]
    gold_labels = [ex['label'] for ex in examples]

    #acc = sum(list(map(lambda v: v[0] == v[1], zip(pred, gold_labels)))) / len(gold_labels)
    #print(acc)
    from sklearn.metrics import confusion_matrix, f1_score
    def f1(y_true, y_pred):
        #y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        confusion = confusion_matrix(y_true, y_pred)
        print("-" * 80 + "Evaluating" + "-" * 80)
        print(confusion)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        return f1_macro, f1_micro

    y_pred = np.array(pred)
    y = np.array(gold_labels)
    f1_macro, f1_micro = np.round(f1(y, y_pred), 5)
    print('lm F1 score: f1_macro = {}, f1_micro = {}'.format(f1_macro, f1_micro))

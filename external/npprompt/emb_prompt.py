import json
import os

from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import GeneralProcessor, AgnewsProcessor, DBpediaProcessor, ImdbProcessor, AmazonProcessor, MicrosoftProcessor
from openprompt.data_utils.glue_dataset import SST2Processor, MNLIProcessor, QNLIProcessor, COLAProcessor, MRPCProcessor, QQPProcessor, RTEProcessor, STSBProcessor
from openprompt.data_utils.huggingface_dataset import YahooAnswersTopicsProcessor
import torch
import argparse
from openprompt import PromptDataLoader
from openprompt.prompts.manual_verbalizer import ManualVerbalizer
from openprompt.prompts.emb_verbalizer import EmbVerbalizer
from openprompt.prompts.manual_template import ManualTemplate
from openprompt import PromptForClassification

from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import load_plm
from sklearn.metrics import classification_report, matthews_corrcoef
from scipy.stats import pearsonr

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--plm_eval_mode", default=True, action="store_true")
parser.add_argument("--model", type=str, default='roberta')
parser.add_argument("--model_name_or_path", default='roberta-large')
parser.add_argument("--result_file", type=str, default="agnews_result.txt")
parser.add_argument("--openprompt_path", type=str, default="./")
parser.add_argument("--verbalizer", type=str, default='ept')
parser.add_argument("--calibration", default=False, action="store_true")
parser.add_argument("--nocut", default=False, action="store_true")
parser.add_argument("--filter", default="tfidf_filter", type=str)
parser.add_argument("--template_id", default=0, type=int)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--dataset", default="agnews", type=str)
parser.add_argument("--select", default=12, type=int)
parser.add_argument("--truncate", default=-1., type=float)
parser.add_argument("--sumprob", default=False, action="store_true")
parser.add_argument("--verbose", default=True, type=bool)
parser.add_argument("--write_filter_record", default=True, action="store_true")
args = parser.parse_args()

set_seed(args.seed)
use_cuda = True
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}
Proc = GeneralProcessor(f"{args.openprompt_path}/datasets/" + args.dataset)
# dataset['train'] = Proc.get_train_examples(f"{args.openprompt_path}/datasets/" + args.dataset)
dataset['test'] = Proc.get_test_examples(f"{args.openprompt_path}/datasets/" + args.dataset)
class_labels = Proc.get_labels()
cutoff = 0.5
max_seq_l = 512
batch_s = 60
num_labels = [i for i in range(len(class_labels))]

# no id, use provided template
mytemplate = ManualTemplate(tokenizer=tokenizer).from_file_(f"{args.openprompt_path}/datasets/" + args.dataset + "/prompt.txt")

if args.verbalizer == "ept":
    myverbalizer = EmbVerbalizer(tokenizer, model=plm, classes=class_labels, candidate_frac=cutoff, max_token_split=args.max_token_split, sumprob=args.sumprob, verbose=args.verbose).from_file(
        select_num=args.select, truncate=args.truncate, dataset_name=args.dataset, path=f'{args.dataset}_{args.model}_cos.pt', tmodel=args.model)
else:
    raise NotImplementedError

prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")
allpreds = []
allprobs = []
alllabels = []
pbar = tqdm(test_dataloader)

all_stat = []

for step, inputs in enumerate(pbar):
    if use_cuda:
        inputs = inputs.cuda()
    stat = prompt_model(inputs)  # batch_size * num_class, 30 * 6
    # all_stat.append(stat)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(stat, dim=-1).cpu().tolist())
    allprobs.append(torch.softmax(stat, dim=-1).cpu())
acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)

retval = args.openprompt_path
INFERENCE_PATH = os.path.join(retval, 'inference')
inference_path = os.path.join(INFERENCE_PATH, args.dataset)
os.system(f"mkdir -p {inference_path}")
json.dump(allpreds, open(f"{inference_path}/eval_labels.json", "w"))
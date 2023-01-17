#!/bin/bash

GPU=$1
PYTHONPATH=python3
BASEPATH="./"
DATASET=$2 #agnews(0) dbpedia(0) imdb(3) amazon(3) yahoo(2) sst2(0) mnli-m(0) mnli-mm(0) cola(0)
TEMPLATEID=0 # 0 1 2 3
SEED=$3 # 145 146 147 148
SHOT=0 # 0 1 10 20
VERBALIZER=ept #
CALIBRATION=""
FILTER=tfidf_filter # none
VERBOSE=1
MODEL=$4 # "roberta"
MODEL_NAME_OR_PATH=$5 # "roberta-large" # "bert-base-uncased"
# RESULTPATH="results_agnews.txt"
OPENPROMPTPATH="."

cd $BASEPATH


rm -f ${DATASET}_${MODEL}_cos.pt

mkdir -p "results/$MODEL_NAME_OR_PATH"

i=$6

CUDA_VISIBLE_DEVICES=$GPU $PYTHONPATH emb_prompt_ProtoCal.py \
        --model $MODEL \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --result_file "results/$MODEL_NAME_OR_PATH/results_$DATASET.txt" \
        --openprompt_path $OPENPROMPTPATH \
        --dataset $DATASET \
        --template_id $TEMPLATEID \
        --seed $SEED \
        --verbose $VERBOSE \
        --verbalizer $VERBALIZER $CALIBRATION \
        --filter $FILTER \
        --select $i

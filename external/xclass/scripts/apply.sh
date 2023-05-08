set -e

gpu=$1
dataset=$2
CUDA_VISIBLE_DEVICES=${gpu} python static_representations.py --dataset_name ${dataset}
CUDA_VISIBLE_DEVICES=${gpu} python class_oriented_document_representations.py --dataset_name ${dataset}
python document_class_alignment.py --dataset_name ${dataset}
python prepare_text_classifer_training.py --dataset_name ${dataset}
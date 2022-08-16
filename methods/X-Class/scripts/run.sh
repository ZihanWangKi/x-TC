set -e

gpu=$1
dataset=$2
seed=$3
CUDA_VISIBLE_DEVICES=${gpu} python static_representations.py --dataset_name ${dataset} --random_state ${seed}
CUDA_VISIBLE_DEVICES=${gpu} python class_oriented_document_representations.py --dataset_name ${dataset} --random_state ${seed}
python document_class_alignment.py --dataset_name ${dataset} --random_state ${seed}
python evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100
python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42
python prepare_text_classifer_training.py --dataset_name ${dataset}
./run_train_text_classifier.sh ${gpu} ${dataset} pca64.clusgmm.bbu-12.mixture-100.42.0.5 ${seed}
python evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100
python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42

set -e

gpu=$1
dataset=$2
seed=$3
lm=$4
layer=$5

echo $lm
CUDA_VISIBLE_DEVICES=${gpu} python static_representations.py --dataset_name ${dataset} --random_state ${seed} --lm_type ${lm} --layer ${layer}
CUDA_VISIBLE_DEVICES=${gpu} python class_oriented_document_representations.py --dataset_name ${dataset} --random_state ${seed} --lm_type ${lm} --layer ${layer}
python document_class_alignment.py --dataset_name ${dataset} --random_state ${seed} --lm_type ${lm}-${layer}
python evaluate.py --dataset ${dataset} --stage Rep --suffix ${lm}-${layer}-mixture-100
python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.${lm}-${layer}.mixture-100.42
python prepare_text_classifer_training.py --dataset_name ${dataset} --suffix pca64.clusgmm.${lm}-${layer}.mixture-100.42
./run_train_text_classifier.sh ${gpu} ${dataset} pca64.clusgmm.${lm}-${layer}.mixture-100.42.0.5 ${seed} ${lm}
python evaluate.py --dataset ${dataset} --stage Rep --suffix ${lm}-${layer}-mixture-100
python evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.${lm}-${layer}.mixture-100.42

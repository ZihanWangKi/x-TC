set -e

gpu=$1
dataset=$2
seed=$3
pca=$4
lm=$5
layer=$6
final_lm=$7
max_len=$8
weight_type=$9
iter=${10}
covariance=${11}

echo $lm
CUDA_VISIBLE_DEVICES=${gpu} python static_representations.py --dataset_name ${dataset} --random_state ${seed} --lm_type ${lm} --layer ${layer}
CUDA_VISIBLE_DEVICES=${gpu} python class_oriented_document_representations.py --dataset_name ${dataset} --random_state ${seed} --lm_type ${lm} --layer ${layer} --attention_mechanism ${weight_type}
python document_class_alignment.py --dataset_name ${dataset} --random_state ${seed} --lm_type ${lm}-${layer} --pca ${pca} --document_repr_type ${weight_type}-100 --iter ${iter} --covariance ${covariance}
python evaluate.py --dataset ${dataset} --stage Rep --suffix ${lm}-${layer}-${weight_type}-100
python evaluate.py --dataset ${dataset} --stage Align --suffix pca${pca}.clusgmm.${lm}-${layer}.${weight_type}-100.${seed}
python prepare_text_classifer_training.py --dataset_name ${dataset} --suffix pca${pca}.clusgmm.${lm}-${layer}.${weight_type}-100.${seed}
./run_train_text_classifier.sh ${gpu} ${dataset} pca${pca}.clusgmm.${lm}-${layer}.${weight_type}-100.${seed}.0.5 ${seed} ${final_lm} ${max_len}
python evaluate.py --dataset ${dataset} --stage Rep --suffix ${lm}-${layer}-${weight_type}-100
python evaluate.py --dataset ${dataset} --stage Align --suffix pca${pca}.clusgmm.${lm}-${layer}.${weight_type}-100.${seed}

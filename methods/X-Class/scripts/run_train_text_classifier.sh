GPU=$1
dataset_name=$2
train_suffix=$3
seed=$4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
model_name_or_path=$5 #bert-base-cased

# this is also defined in utils.py, make sure to change both when changing.
output_dir=../models/${model_name_or_path}_${train_suffix}

seq_length=512
if [ ${model_name_or_path} == "blu" ] || [ ${model_name_or_path} == "blc" ]; then
    seq_length=128
fi

CUDA_VISIBLE_DEVICES=$GPU python train_text_classifier.py \
  --model_name_or_path ${model_name_or_path} \
  --task_name ${dataset_name} \
  --train_suffix ${train_suffix} \
  --test_suffix "test" \
  --output_dir ${output_dir} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length ${seq_length} \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --logging_steps 100000 \
  --save_steps -1 \
  --seed ${seed} \
  --overwrite_output_dir \
  --overwrite_cache

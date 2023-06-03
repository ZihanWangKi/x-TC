dataset_name=$1
train_suffix=$2
exp_name=$3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
model_name_or_path=bert-base-cased
data_dir=${exp_name}/data/datasets
output_dir=${exp_name}/inference/${dataset_name}

python train_text_classifier.py \
  --data_dir ${data_dir}\
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
  --max_seq_length 512 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --logging_steps 100000 \
  --save_steps -1 \
  --overwrite_output_dir \
  --overwrite_cache

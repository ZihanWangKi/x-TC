gpu=$1

MASTER_PORT=18672 taskset -c 40-60 python run.py  --method ClassKG --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm roberta-base"
MASTER_PORT=18672 taskset -c 40-60 python run.py  --method ClassKG --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm roberta-large"
MASTER_PORT=18672 taskset -c 40-60 python run.py  --method ClassKG --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=18672 taskset -c 40-60 python run.py  --method ClassKG --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-base-uncased --clustering"

MASTER_PORT=18672 taskset -c 40-60 python run.py  --method ClassKG --dataset 20News --class_names_file ./default_class_names/20.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--lm bert-base-uncased --clustering"


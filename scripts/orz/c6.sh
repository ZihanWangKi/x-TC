gpu=$1

MASTER_PORT=18676 taskset -c 20-100 python run.py  --method ClassKG --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--lm roberta-large"
MASTER_PORT=18676 taskset -c 20-100 python run.py  --method ClassKG --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--lm roberta-base"
MASTER_PORT=18676 taskset -c 20-100 python run.py  --method ClassKG --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=18676 taskset -c 20-100 python run.py  --method ClassKG --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--clustering "
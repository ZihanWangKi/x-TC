gpu=$1

MASTER_PORT=10672 taskset -c 60-90 python run.py  --method ClassKG --dataset NYT-fine --class_names_file ./default_class_names/nyt-fine.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "clustering "
MASTER_PORT=10672 taskset -c 60-90 python run.py  --method ClassKG --dataset NYT-fine --class_names_file ./default_class_names/nyt-fine.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
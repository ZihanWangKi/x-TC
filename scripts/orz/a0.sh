gpu=$1

MASTER_PORT=10670 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./review/3.txt --split_ratio 0.8 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=10670 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./review/4.txt --split_ratio 0.8 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=10670 taskset -c 60-90 python run.py  --method ClassKG --dataset 20News-fine --class_names_file ./default_class_names/20-fine.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "clustering"
gpu=$1

MASTER_PORT=18674 taskset -c 80-100 python run.py  --method ClassKG --dataset 20News --class_names_file ./default_class_names/20.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"


MASTER_PORT=18674 taskset -c 80-100 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.8 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--clustering"
MASTER_PORT=18674 taskset -c 80-100 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.8 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=18674 taskset -c 80-100 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.8 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm roberta-base"
MASTER_PORT=18674 taskset -c 80-100 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.8 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm roberta-large"

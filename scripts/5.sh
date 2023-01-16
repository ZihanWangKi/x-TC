gpu=$1
MASTER_PORT=13177 taskset -c 90-120 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./dbpedia/4.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
MASTER_PORT=17177 taskset -c 90-120 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./dbpedia/4.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-large-uncased"

MASTER_PORT=18177 taskset -c 90-120 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
MASTER_PORT=18177 taskset -c 90-120 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm roberta-base"
MASTER_PORT=18177 taskset -c 90-120 python run.py  --method ClassKG --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm roberta-large"

MASTER_PORT=18177 taskset -c 90-120 python run.py  --method ClassKG --dataset NYT-Topics --class_names_file ./default_class_names/nyt-top.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--lm roberta-base"
MASTER_PORT=18177 taskset -c 90-120 python run.py  --method ClassKG --dataset NYT-Topics --class_names_file ./default_class_names/nyt-top.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--lm roberta-large"
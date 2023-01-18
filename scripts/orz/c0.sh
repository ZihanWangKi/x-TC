gpu=$1

MASTER_PORT=18670 taskset -c 0-20 python run.py  --method ClassKG --dataset imdb --class_names_file ./default_class_names/review.txt --split_ratio 0.7 --class_names --train_size 0.2 --test_size 0.2 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=18670 taskset -c 0-20 python run.py  --method ClassKG --dataset imdb --class_names_file ./default_class_names/review.txt --split_ratio 0.7 --class_names --train_size 0.2 --test_size 0.2 --gpu ${gpu} --suffix "--clustering"
MASTER_PORT=18670 taskset -c 0-20 python run.py  --method ClassKG --dataset yelp_review_full --class_names_file ./default_class_names/review_5.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=18670 taskset -c 0-20 python run.py  --method ClassKG --dataset yelp_review_full --class_names_file ./default_class_names/review_5.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--clustering"

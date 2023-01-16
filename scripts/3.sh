gpu=$1
#MASTER_PORT=10877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./imdb/1.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
#MASTER_PORT=11877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./imdb/2.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
#MASTER_PORT=12877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./review/3.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
#MASTER_PORT=13877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./review/4.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
#MASTER_PORT=14877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./review/3.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
#MASTER_PORT=15877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./review/4.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm bert-large-uncased"

MASTER_PORT=16877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./default_class_names/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm roberta-base"
MASTER_PORT=16877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./default_class_names/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm roberta-large"

MASTER_PORT=17877 taskset -c 60-90 python run.py  --method ClassKG --dataset imdb --class_names_file ./default_class_names/review.txt --split_ratio 0.7 --class_names --train_size 0.2 --test_size 0.2 --gpu ${gpu} --suffix "--lm roberta-base"
MASTER_PORT=17877 taskset -c 60-90 python run.py  --method ClassKG --dataset imdb --class_names_file ./default_class_names/review.txt --split_ratio 0.7 --class_names --train_size 0.2 --test_size 0.2 --gpu ${gpu} --suffix "--lm roberta-large"

MASTER_PORT=18877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_review_full --class_names_file ./default_class_names/review_5.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm roberta-base"
MASTER_PORT=18877 taskset -c 60-90 python run.py  --method ClassKG --dataset yelp_review_full --class_names_file ./default_class_names/review_5.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--lm roberta-large"

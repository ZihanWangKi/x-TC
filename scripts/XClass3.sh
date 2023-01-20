gpu=$1

taskset -c 60-90 python run.py  --dataset yelp_review_full --class_names_file ./default_class_names/review_5.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"
taskset -c 60-90 python run.py  --dataset imdb --class_names_file ./default_class_names/review.txt --split_ratio 0.7 --class_names --train_size 0.2 --test_size 0.2 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"


taskset -c 60-90 python run.py --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 roberta-base 12 roberta-base 128 mixture gmm"
taskset -c 60-90 python run.py --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "100 roberta-large 24 roberta-large 128 mixture gmm"

taskset -c 60-90 python run.py  --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia_14.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 roberta-base 12 roberta-base 128 mixture gmm"
taskset -c 60-90 python run.py  --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia_14.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "100 roberta-large 24 roberta-large 128 mixture gmm"

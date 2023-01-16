gpu=$1
taskset -c 30-60 python run.py --method LOTClass --dataset imdb --class_names_file ./default_class_names/review.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.2 --test_size 0.2 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-base"
taskset -c 30-60 python run.py --method LOTClass --dataset imdb --class_names_file ./default_class_names/review.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.2 --test_size 0.2 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-large"

taskset -c 30-60 python run.py --method LOTClass --dataset yelp_review_full --class_names_file ./default_class_names/review_5.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-base"
taskset -c 30-60 python run.py --method LOTClass --dataset yelp_review_full --class_names_file ./default_class_names/review_5.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-large"

taskset -c 30-60 python run.py --method LOTClass --dataset NYT --class_names_file ./default_class_names/nyt.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-base"
taskset -c 30-60 python run.py --method LOTClass --dataset NYT --class_names_file ./default_class_names/nyt.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-large"

taskset -c 30-60 python run.py --method LOTClass --dataset NYT-fine --class_names_file ./default_class_names/nyt-fine.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-base"
taskset -c 30-60 python run.py --method LOTClass --dataset NYT-fine --class_names_file ./default_class_names/nyt-fine.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-large"

taskset -c 30-60 python run.py --method LOTClass --dataset NYT-Topics --class_names_file ./default_class_names/nyt-top.txt --prompt_file ./prompts/topic.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-base"
taskset -c 30-60 python run.py --method LOTClass --dataset NYT-Topics --class_names_file ./default_class_names/nyt-top.txt --prompt_file ./prompts/topic.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-large"

taskset -c 30-60 python run.py --method LOTClass --dataset NYT-Locations --class_names_file ./default_class_names/nyt-loc.txt --prompt_file ./prompts/loc.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-base"
taskset -c 30-60 python run.py --method LOTClass --dataset NYT-Locations --class_names_file ./default_class_names/nyt-loc.txt --prompt_file ./prompts/loc.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-large"


taskset -c 30-60 python run.py --method LOTClass --dataset 20News-fine --class_names_file ./default_class_names/20-fine.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-base"
taskset -c 30-60 python run.py --method LOTClass --dataset 20News-fine --class_names_file ./default_class_names/20-fine.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--top_pred_num 100 --match_threshold 10 --max_len 128 --pretrained_lm roberta-large"

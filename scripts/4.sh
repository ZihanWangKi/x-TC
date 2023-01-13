MASTER_PORT=10877 taskset -c 30-60 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./imdb/1.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --suffix "--lm bert-large-uncased"
MASTER_PORT=11877 taskset -c 30-60 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./imdb/2.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --suffix "--lm bert-large-uncased"
MASTER_PORT=12877 taskset -c 30-60 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./imdb/3.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --suffix "--lm bert-large-uncased"
MASTER_PORT=13877 taskset -c 30-60 python run.py  --method ClassKG --dataset yelp_polarity --class_names_file ./imdb/4.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --suffix "--lm bert-large-uncased"

#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
gpu=$1

MASTER_PORT=18677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"

MASTER_PORT=18677 taskset -c 90-120 python run.py  --method ClassKG --dataset NYT-Locations --class_names_file ./default_class_names/nyt-loc.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=18677 taskset -c 90-120 python run.py  --method ClassKG --dataset 20News --class_names_file ./default_class_names/20.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"

#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
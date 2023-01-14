gpu=$1
MASTER_PORT=10677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./agnews/1.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
MASTER_PORT=11677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./agnews/2.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
MASTER_PORT=12677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./agnews/3.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-base-uncased"
MASTER_PORT=13677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./agnews/4.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-base-uncased"

MASTER_PORT=14677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./agnews/1.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=15677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./agnews/2.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=16677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./agnews/3.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=17677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./agnews/4.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"

MASTER_PORT=18677 taskset -c 90-120 python run.py  --method ClassKG --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
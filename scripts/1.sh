taskset -c 30-60 python run.py  --method gpt2-small --dataset NYT --class_names_file ./nyt/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 5 --prompt --suffix "--iter 100"
taskset -c 30-60 python run.py  --method gpt2-small --dataset NYT --class_names_file ./nyt/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 5 --prompt --suffix "--iter 100"
taskset -c 30-60 python run.py  --method gpt2-small --dataset NYT --class_names_file ./nyt/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 5 --prompt --suffix "--iter 100"
taskset -c 30-60 python run.py  --method gpt2-small --dataset NYT --class_names_file ./nyt/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 5 --prompt --suffix "--iter 100"

taskset -c 30-60 python run.py  --method gpt2-small --dataset NYT --class_names_file ./nyt/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 5 --prompt --suffix "--iter 100" --additional_method ProtoCal
taskset -c 30-60 python run.py  --method gpt2-small --dataset NYT --class_names_file ./nyt/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 5 --prompt --suffix "--iter 100" --additional_method ProtoCal
taskset -c 30-60 python run.py  --method gpt2-small --dataset NYT --class_names_file ./nyt/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 5 --prompt --suffix "--iter 100" --additional_method ProtoCal
taskset -c 30-60 python run.py  --method gpt2-small --dataset NYT --class_names_file ./nyt/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 5 --prompt --suffix "--iter 100" --additional_method ProtoCal

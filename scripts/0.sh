taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/1.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/1.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"

taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/1.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/1.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal


taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/2.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/2.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"

taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/2.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/2.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal



taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/3.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/3.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"

taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/3.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/3.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal


taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/4.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"
taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/4.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100"

taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/4.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/ag/4.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

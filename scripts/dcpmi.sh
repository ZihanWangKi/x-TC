taskset -c 60-90 python run.py  --method gpt2-medium --dataset 20News --class_names_file ./default_class_names/20.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 1 --suffix "--iter 100 --dcpmi" --additional_method ProtoCal --prompt --prompt_file ./prompts/topic.txt
taskset -c 60-90 python run.py  --method gpt2-small --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 1 --suffix "--iter 100 --dcpmi" --additional_method ProtoCal --prompt --prompt_file ./prompts/topic.txt
taskset -c 60-90 python run.py  --method gpt2-medium --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 1 --suffix "--iter 100 --dcpmi" --additional_method ProtoCal --prompt --prompt_file ./prompts/topic.txt
taskset -c 60-90 python run.py  --method gpt2-small --dataset ag_news --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 1 --suffix "--iter 100 --dcpmi" --additional_method ProtoCal --prompt --prompt_file ./prompts/topic.txt
taskset -c 60-90 python run.py  --method gpt2-medium --dataset ag_news --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 1 --suffix "--iter 100 --dcpmi" --additional_method ProtoCal --prompt --prompt_file ./prompts/topic.txt

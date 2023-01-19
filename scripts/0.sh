
taskset -c 60-90 python run.py  --method gpt2-small --dataset NYT-Topics --class_names_file ./default_class_names/nyt-top.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu 6 --suffix "--iter 100 --dcpmi" --additional_method ProtoCal --prompt --prompt_file ./prompts/topic.txt
taskset -c 60-90 python run.py  --method gpt2-medium --dataset NYT-Topics --class_names_file ./default_class_names/nyt-top.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu 6 --suffix "--iter 100 --dcpmi" --additional_method ProtoCal --prompt --prompt_file ./prompts/topic.txt

taskset -c 60-90 python run.py  --method gpt2-medium --dataset NYT-fine --class_names_file ./default_class_names/nyt-fine.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu 6 --suffix "--iter 100 --dcpmi" --additional_method ProtoCal --prompt --prompt_file ./prompts/topic.txt

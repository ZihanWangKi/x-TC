gpu=$1
taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./default_class_names/review.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./default_class_names/review.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

taskset -c 30-60 python run.py  --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
taskset -c 30-60 python run.py  --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

taskset -c 30-60 python run.py  --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
taskset -c 30-60 python run.py  --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

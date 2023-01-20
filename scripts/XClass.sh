gpu=$1
#taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./agnews/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
#taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./agnews/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

#taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./agnews/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
#taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./agnews/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

#taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./agnews/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
#taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./agnews/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

#taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./agnews/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
#taskset -c 30-60 python run.py  --dataset ag_news --class_names_file ./agnews/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"


#taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./review/1.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
#taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./review/1.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

#taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./review/2.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
#taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./review/2.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

#taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./review/3.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
#taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./review/3.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"

#taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./review/4.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture 0 tied"
#taskset -c 30-60 python run.py  --dataset yelp_polarity --class_names_file ./review/4.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture 0 tied"


#taskset -c 30-60 python run.py  --dataset NYT --class_names_file ./nyt/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture gmm"
#taskset -c 30-60 python run.py  --dataset NYT --class_names_file ./nyt/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture gmm"

#taskset -c 30-60 python run.py  --dataset NYT --class_names_file ./nyt/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture gmm"
#taskset -c 30-60 python run.py  --dataset NYT --class_names_file ./nyt/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture gmm"

#taskset -c 30-60 python run.py  --dataset NYT --class_names_file ./nyt/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture gmm"
#taskset -c 30-60 python run.py  --dataset NYT --class_names_file ./nyt/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture gmm"

#taskset -c 30-60 python run.py  --dataset NYT --class_names_file ./nyt/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture gmm"
#taskset -c 30-60 python run.py  --dataset NYT --class_names_file ./nyt/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "100 blu 24 blc 128 mixture gmm"

taskset -c 30-60 python run.py  --dataset NYT-Topics --class_names_file ./default_class_names/nyt-top.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"
taskset -c 30-60 python run.py  --dataset NYT-Locations --class_names_file ./default_class_names/nyt-loc.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"
taskset -c 30-60 python run.py  --dataset NYT-fine --class_names_file ./default_class_names/nyt-fine.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"

#taskset -c 30-60 python run.py  --class_names_file ./default_class_names/ag.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"

#taskset -c 30-60 python run.py  --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"

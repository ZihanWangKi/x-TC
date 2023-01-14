#taskset -c 30-60 python run.py  --method gpt2-small --dataset yelp_polarity --class_names_file ./review/1.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method gpt2-medium --dataset yelp_polarity --class_names_file ./review/1.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

#taskset -c 30-60 python run.py  --method gpt2-small --dataset yelp_polarity --class_names_file ./review/2.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method gpt2-medium --dataset yelp_polarity --class_names_file ./review/2.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

taskset -c 30-60 python run.py  --method gpt2-small --dataset yelp_polarity --class_names_file ./review/3.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-small --dataset yelp_polarity --class_names_file ./review/3.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

taskset -c 30-60 python run.py  --method gpt2-medium --dataset yelp_polarity --class_names_file ./review/3.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium --dataset yelp_polarity --class_names_file ./review/3.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

taskset -c 30-60 python run.py  --method gpt2-small --dataset yelp_polarity --class_names_file ./review/4.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-small --dataset yelp_polarity --class_names_file ./review/4.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

taskset -c 30-60 python run.py  --method gpt2-medium --dataset yelp_polarity --class_names_file ./review/4.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100"
#taskset -c 30-60 python run.py  --method gpt2-medium --dataset yelp_polarity --class_names_file ./review/4.txt --prompt_file ./prompts/review.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

#taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./agnews/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/1.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

#taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./agnews/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/2.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

#taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./agnews/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/3.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

#taskset -c 30-60 python run.py  --method gpt2-small  --class_names_file ./agnews/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method gpt2-medium  --class_names_file ./agnews/4.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu 3 --prompt --suffix "--iter 100" --additional_method ProtoCal

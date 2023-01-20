gpu=$1

#taskset -c 30-60 python run.py  --method BERTprompt --dataset 20News --class_names_file ./default_class_names/20.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --prompt --suffix "--lm_type bbc --add_mask --dcpmi"
#taskset -c 30-60 python run.py  --method BERTprompt --dataset yelp_polarity --class_names_file ./default_class_names/review.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --prompt --suffix "--lm_type bbc --add_mask --dcpmi"


taskset -c 30-60 python run.py  --method BERTprompt --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --prompt --suffix "--lm_type bbc --add_mask --dcpmi"
taskset -c 30-60 python run.py  --method BERTprompt --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --prompt --suffix "--lm_type blc --add_mask --dcpmi"
taskset -c 30-60 python run.py  --method BERTprompt --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --prompt --suffix "--lm_type roberta-base --add_mask --dcpmi"
taskset -c 30-60 python run.py  --method BERTprompt --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --prompt --suffix "--lm_type roberta-large --add_mask --dcpmi"
taskset -c 30-60 python run.py  --method BERTprompt --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --prompt --suffix "--lm_type bart-base --add_mask --dcpmi"
taskset -c 30-60 python run.py  --method BERTprompt --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --prompt --suffix "--lm_type bart-large --add_mask --dcpmi"
#taskset -c 30-60 python run.py  --method BERTprompt --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 10. --gpu ${gpu} --prompt --suffix "--lm_type electra-base --add_mask --dcpmi"
#taskset -c 30-60 python run.py  --method BERTprompt --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 10. --gpu ${gpu} --prompt --suffix "--lm_type electra-large --add_mask --dcpmi"


#taskset -c 30-60 python run.py  --method BERTprompt --dataset ag_news --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --prompt --suffix "--lm_type bbc --add_mask --dcpmi" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method BERTprompt --dataset ag_news --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --prompt --suffix "--lm_type blc --add_mask --dcpmi" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method BERTprompt --dataset ag_news --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --prompt --suffix "--lm_type roberta-base --add_mask --dcpmi" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method BERTprompt --dataset ag_news --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --prompt --suffix "--lm_type roberta-large --add_mask --dcpmi" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method BERTprompt --dataset ag_news --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --prompt --suffix "--lm_type bart-base --add_mask --dcpmi" --additional_method ProtoCal
#taskset -c 30-60 python run.py  --method BERTprompt --dataset ag_news --class_names_file ./default_class_names/ag.txt --prompt_file ./prompts/topic.txt --split_ratio 0.7 --class_names --train_size 0.05 --test_size 1 --gpu ${gpu} --prompt --suffix "--lm_type bart-large --add_mask --dcpmi" --additional_method ProtoCal

gpu=$1
taskset -c 60-90 python run.py  --dataset 20News-fine --class_names_file ./default_class_names/20-fine.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"
taskset -c 60-90 python run.py  --dataset 20News --class_names_file ./default_class_names/20.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"

taskset -c 60-90 python run.py  --dataset NYT --class_names_file ./default_class_names/nyt.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"
taskset -c 60-90 python run.py  --dataset dbpedia_14 --text_name title content --class_names_file ./default_class_names/dbpedia.txt --split_ratio 0.7 --class_names --train_size 0.01 --test_size 0.1 --gpu ${gpu} --suffix "64 bbu 12 bbc 128 mixture none"


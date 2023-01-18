gpu=$1

MASTER_PORT=18673 taskset -c 60-80 python run.py  --method ClassKG --dataset NYT_Topics --class_names_file ./default_class_names/nyt-top.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--clustering "
MASTER_PORT=18673 taskset -c 60-80 python run.py  --method ClassKG --dataset NYT-Topics --class_names_file ./default_class_names/nyt-top.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"
MASTER_PORT=18673 taskset -c 60-80 python run.py  --method ClassKG --dataset NYT-Locations --class_names_file ./default_class_names/nyt-loc.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--clustering "
MASTER_PORT=18673 taskset -c 60-80 python run.py  --method ClassKG --dataset NYT-Locations --class_names_file ./default_class_names/nyt-loc.txt --split_ratio 0.8 --class_names --train_size 0.2 --test_size 1 --gpu ${gpu} --suffix "--lm bert-large-uncased"

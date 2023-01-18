gpu=$1

MASTER_PORT=10671 taskset -c 40-90 python run.py  --method ClassKG --dataset NYT --class_names_file ./nyt/1.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu}
MASTER_PORT=10671 taskset -c 40-90 python run.py  --method ClassKG --dataset NYT --class_names_file ./nyt/2.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu}
MASTER_PORT=10671 taskset -c 40-90 python run.py  --method ClassKG --dataset NYT --class_names_file ./nyt/3.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu}
MASTER_PORT=10671 taskset -c 40-90 python run.py  --method ClassKG --dataset NYT --class_names_file ./nyt/4.txt --split_ratio 0.7 --class_names --train_size 0.5 --test_size 1 --gpu ${gpu}

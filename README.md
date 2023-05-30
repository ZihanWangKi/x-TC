# A Benchmark on Extremely Weakly Supervised Text Classification: Reconcile Seed Matching and Prompting Approaches

This repo contains the data and the code for the [paper](https://arxiv.org/abs/2305.12749).

## Requirements
The python version we used is 3.8, also, you need to install torch. The other requirements are listed in `requirements.txt`.

## Data & Methods

The datasets can be accessed at `data/`.

The methods are implemented in `external/` (we copied the core parts for each method, modified to suit taking in different hyperparameters if necessary) and we provide classes to invoke them in `methods/`. 

We benchmarked the following methods:
- Prompt
  - Prompting
  - Prompting + DC-PMI ([github](https://github.com/peterwestuw/surface-form-competition))
  - Prompting + ProtoCal ([paper](https://arxiv.org/abs/2205.10183))
- Seed Matching
  - LoTClass ([github](https://github.com/yumeng5/LOTClass))
  - XClass ([github](https://github.com/ZihanWangKi/XClass))
  - ClassKG ([github](https://github.com/zhanglu-cst/ClassKG))
  - NPPrompt ([github](https://arxiv.org/abs/2212.06950))

To test a model (e.g., prompting) on a dataset (e.g., NYT-Topics), you may run
```
method_name=prompt_gpt
lm_name=gpt2
data_name=NYT-Topics

CUDA_VISIBLE_DEVICES=${gpu} python run.py \
    --method ${method_name} \
    --base_model ${lm_name} \
    --hyperparameter_file_path methods/hyperparameters/${method_name}.json \
    --data ${data_name} \
    --label_names_file_name data/${data_name}/label_names.txt \
    --prompt_file_name data/${data_name}/prompt.txt
```

The performances on the datasets, their behaviors when using different label names, instructions, pre-trained language models, can be found in our [paper](https://arxiv.org/abs/2305.12749).


## Citation
If you find this repo useful, please cite our paper:
```
@article{wang2023benchmark,
  title={A Benchmark on Extremely Weakly Supervised Text Classification: Reconcile Seed Matching and Prompting Approaches},
  author={Wang, Zihan and Wang, Tianle and Mekala, Dheeraj and Shang, Jingbo},
  journal={arXiv preprint arXiv:2305.12749},
  year={2023}
}
```
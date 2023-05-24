# Datasets for x-TC

### Format
1. All datasets should be stored in the folder prefixed with its name, and contain three files:
    - `train_text.txt`: the texts for unsupervised training
    - `label_names.txt`: the label names for testing
    - `test_text.txt`: the texts for testing
    - `test_label.txt`: the labels for testing
    - (optional) `train_label.txt`: the labels for unsupervised training, can be used for verification purposes
    - (optional) `prompt.txt`: the textual prompt for prompting, this can also be overwritten in method arguments

2. We provide a sample dataset from AG's News Topic Classification dataset. And the datasets referenced in our paper have been pre-processed and provided in a ready-to-use format. For additional datasets referenced in our paper and available on HuggingFace, we provide a Python data processor `dataset_processor.py` with instructions to access these resources. 
   1. ag_news
   ```
   python dataset_processor.py --dataset ag_news --train_size 0.05 --train_label
   ```
   2. imdb
   ```
   python dataset_processor.py --dataset imdb --train_size 0.2 --test_size 0.2 --train_label
   ```
   3. yelp_polarity
   ```
   python dataset_processor.py --dataset yelp_polarity --train_size 0.01 --test_size 0.1 --train_label
   ```
   4. yelp_review_full
   ```
   python dataset_processor.py --dataset yelp_review_full --train_size 0.01 --test_size 0.1 --train_label
   ```
   5. dbpedia_14
   ```
   python dataset_processor.py --text title content --dataset dbpedia_14 --train_size 0.01 --test_size 0.1 --train_label
   ```

This data processor can also be used to process other new datasets on HuggingFace.
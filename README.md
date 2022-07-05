# x-TC
Standing for eXtremely weakly supervsied Text Classification, this project will be for benchmarking a series of text classifcation methods on a collection of (standarized) datasets, with the goal of evaluating their performances under the extremely weakly supervised scenario.

## Datasets
We consider the following datasets:
- AG News (link...)
- and more ...

The datasets should be standarized in the following ways:
- Preprocessing (removing urls, corrupted text, allowing only ascii characters?)
- Data Splits
    - Which part contains the evaluation split. (For a single dataset, we can even create several splits for different class balancing ratios)
    - Training/Supervision set (transductive/inductive), where the supervision (can be unlabelled text) comes from.
    - Naming Convention

The datasets should cover a range of different class criterion:
- Topics (science, sports, business)
- Location (US, France, ...)
- Sentiment (Happy, Sad)
- more?

The datasets should come from different domain:
- News
- Reviews
- Wikipedia 
- ...

## Methods
We consider the following methods:
- Pseudo Labeling
    - WeSTClass
    - ConWea
    - LotClass
    - XClass
    - more?
- Prompt Learning
    - gpt3
    - calibrations (dm-pmi, contextual, prototypical)
    - meta trained (metaicl, t0)
- Self Training as Post Processing

## Supervision
The main experiment is under extremely weakly supervised setting with the names of the classes as the only supervision.
We also consider two slightly more supervision scenario
- seed words, apart from the class names, additional key words for each class is given.
- few labelled documents, apart from the class names, additional labelled documents for each class is given.
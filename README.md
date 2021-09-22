# Learning Data Augmentation Schedules for NLP

<b><i>This code is made available for both reproducibility purposes and to allow for further research on the topic of data augmentation for NLP</b></i>

<b><i>The PBA part of the code is based on https://github.com/arcelien/pba</b></i>
<b><i>and some of the rest comes from other publicly available repositories (see specific files for details)</b></i>


### Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Reproduce Results](#reproduce-results)
4. [Run PBA Search](#run-pba-search)
5. [Run PBA Search and Evaluation on New Dataset](#run-pba-search-eval-new-dataset)

### Introduction

This repository contains code for the work "Learning Data Augmentation Schedules for NLP" in TensorFlow and Python.

### Getting Started

####  Install requirements

The code was tested on Python 3.6.10.

```shell
pip install -r requirements.txt
```

#### Download SST-2 and MNLI datasets

Download the SST-2 and MNLI datasets from GLUE and paste the downloaded files in `datasets/glue_data/SST-2` and `datasets/glue_data/MNLI` respectively.

#### Download BERT base uncased model

Download the BERT base uncase model from BERT and paste the files `bert_config.json`, `bert_model.ckpt.data-00000-of-00001`, `bert_model.ckpt.index`, `bert_model.ckpt.meta` and `vocab.txt` in `datasets/pretrained_models/bert_base`.

#### Download precomputed augmentated data

Download the pre-generated augmentated data for contextual augmentation using the link `https://drive.google.com/file/d/1XYnp2MsqTZvm0nl8UydthYneoP0YJtGv/view?usp=sharing`. 

### Reproduce Results

Scripts to reproduce results with augmenation reported in the paper are located in `scripts/eval_glue_*.sh`. And results without augmentation are located in `scripts/no_aug_glue_*.sh`. One argument, the dataset name (choices are `sst2` or `mnli_mm`), is required for all of the scripts. Hyperparamaters are also located inside each script file. The hyperparameter `--cpu` should be modified to fit the number of available cpus.

For example, to reproduce the SST-2 results with augmentation and dataset size N=3000:

```shell
bash scripts/eval_glue_3000.sh sst2
```

To reproduce the MNLI results with augmentation and N=1500:

```shell
bash scripts/eval_glue_1500.sh mnli_mm
```

To reproduce the results without augmentation, for example on SST-2 with dataset size N=9000:

```shell
bash scripts/no_aug_glue_FULL.sh sst2
```


### Run PBA Search

Run PBA search with the file `scripts/search_*.sh`. One argument, the dataset name, is required. Choices are `sst2` or `mnli_mm`. The star replaces the dataset size (choices are `1500`, `2000`, `3000`, `4000`, `FULL`). For example to run a search for N=2000 on MNLI:


```shell
CUDA_VISIBLE_DEVICES=0 bash scripts/search_2000.sh mnli_mm
```

The resulting schedules used in search can be retrieved from the Ray result directory, and the log files can be converted into policy schedules with the `main()` function in `pba/utils.py`. The name of the experiment (name of the folder that contains the search results in `pba/results`) needs to be manually given at the beginning of the `main()` using the variable `exp_name`. Similarly, the variable `task_name` is used to indicate the name of the dataset.


### Run PBA Search and Evaluation on New Dataset

Details will follow


### Citation
If you use this code in your research, please cite:

```
@inproceedings{chopard2021learning,
  title     = {Learning Data Augmentation Schedules for Natural Language Processing},
  author    = {Daphn{\'e} Chopard and
               Matthias S. Treder and
               Irena Spasi{\'c}},
  maintitle = {EMNLP Workshop},
  booktitle = {Proceedings of the Second Workshop on Insights from Negative Results in NLP},
  year      = {2021}
}
```

#!/bin/bash
export PYTHONPATH="$(pwd)"
echo $PYTHONPATH

sst2_search() {
    local_dir="$PWD/results/"
    data_path="$PWD/datasets/"

    python3.6 pba/search.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name bert --dataset sst2 --expsize "1500" \
    --train_size 250 --val_size 1250 \
    --checkpoint_freq 0 --development --max_seq_length 128 \
    --name "sst2_search_all_augm_1500" --gpu 1 --cpu 6 \
    --num_samples 16 --perturbation_interval 1 --epochs 48 \
    --explore default_explore_fct --aug_policy default_aug_policy \
    --lr 1e-5 --wd 0.0001 --bs 32
}

mnlimm_search() {
    local_dir="$PWD/results/"
    data_path="$PWD/datasets/"

    python3.6 pba/search.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name bert --dataset mnli_mm --expsize "1500" \
    --train_size 250 --val_size 1250 \
    --checkpoint_freq 0 --development --max_seq_length 128 \
    --name "mnli_mm_search_all_augm_1500" --gpu 1 --cpu 6 \
    --num_samples 16 --perturbation_interval 1 --epochs 48 \
    --explore default_explore_fct --aug_policy default_aug_policy \
    --lr 5e-5 --wd 0.0001 --bs 32
}


if [ "$1" = "sst2" ]; then
    sst2_search
elif [ "$1" = "mnli_mm" ]; then
    mnlimm_search
else
    echo "invalid args"
fi

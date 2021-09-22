#!/bin/bash
export PYTHONPATH="$(pwd)"


eval_no_aug_mnli_mm() {
  local_dir="$PWD/results/"
  data_path="$PWD/datasets/"

  size=3000
  dataset="mnli_mm"

  python3.6 pba/train.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name bert --dataset "$dataset" --expsize "3000" \
    --train_size "$size" --val_size 0 \
    --checkpoint_freq 0 --gpu 1 --cpu 4 \
    --epochs 48 --development --max_seq_length 128 \
    --no_aug --name "mnli_mm_no_aug_eval_3000" \
    --lr 2e-5 --wd 0.1 --bs 32
}

eval_no_aug_sst2() {
  local_dir="$PWD/results/"
  data_path="$PWD/datasets/"

  size=3000
  dataset="sst2"

  python3.6 pba/train.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name bert --dataset "$dataset" --expsize "3000" \
    --train_size "$size" --val_size 0 \
    --checkpoint_freq 0 --gpu 1 --cpu 4 \
    --epochs 48 --development --max_seq_length 128 \
    --no_aug --name "sst2_no_aug_eval_3000" \
    --lr 1e-5 --wd 0.1 --bs 32
}



if [ "$1" = "mnli_mm" ]; then
   eval_no_aug_mnli_mm
elif [ "$1" = "sst2" ]; then
   eval_no_aug_sst2
else
    echo "invalid args"
fi


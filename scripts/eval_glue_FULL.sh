#!/bin/bash
export PYTHONPATH="$(pwd)"

# args: [model name] [lr] [wd]
eval_sst2() {
  hp_policy="$PWD/schedules/sst2_search_all_augm_FULL.txt"
  local_dir="$PWD/results/"
  data_path="$PWD/datasets/"
  name="eval_sst2_schedule_FULL"
  size=9000
  dataset="sst2"

  python3.6 pba/train.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name bert --dataset "$dataset" --expsize "FULL" \
    --train_size "$size" --val_size 0 --development \
    --checkpoint_freq 0 --gpu 1 --cpu 4 \
    --use_hp_policy --hp_policy "$hp_policy" \
    --hp_policy_epochs 48 --epochs 48 \
    --aug_policy default_aug_policy --name "$name" \
    --lr "$1" --bs 32
}

eval_mnlimm() {
  hp_policy="$PWD/schedules/mnli_mm_search_all_augm_FULL.txt"
  local_dir="$PWD/results/"
  data_path="$PWD/datasets/"
  name="eval_mnli_schedule_FULL"
  size=9000
  dataset="mnli_mm"

  python3.6 pba/train.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name bert --dataset "$dataset" --expsize "FULL" \
    --train_size "$size" --val_size 0 --development \
    --checkpoint_freq 0 --gpu 1 --cpu 4 \
    --use_hp_policy --hp_policy "$hp_policy" \
    --hp_policy_epochs 48 --epochs 48 \
    --aug_policy default_aug_policy --name "$name" \
    --lr "$1" --bs 32
}

if [ "$@" = "sst2" ]; then
  eval_sst2 1e-5
elif [ "$@" = "mnli_mm" ]; then
  eval_mnlimm 5e-5
else
  echo "invalid args"
fi

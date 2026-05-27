#!/bin/bash
# Train CEN on MSVD-CTN, fine-tuning from a checkpoint trained on MSRVTT-CTN.

set -euo pipefail

N_GPU=1
N_THREAD=8

DATA_PATH=/path/to/MSVD_CTN_data.json
CKPT_ROOT=/path/to/checkpoints
INIT_MODEL_PATH=/path/to/msrvtt_pretrained/pytorch_model.bin
FEATURES_PATH_CAUSE=/path/to/MSVD_cause_features.pickle
FEATURES_PATH_EFFECT=/path/to/MSVD_effect_features.pickle

python -m torch.distributed.launch --nproc_per_node=${N_GPU} \
  train_msvd.py \
  --do_train \
  --num_thread_reader ${N_THREAD} \
  --epochs 50 \
  --batch_size 64 --batch_size_val 128 \
  --gradient_accumulation_steps 1 \
  --n_display 50 \
  --data_path ${DATA_PATH} \
  --features_path_cause ${FEATURES_PATH_CAUSE} \
  --features_path_effect ${FEATURES_PATH_EFFECT} \
  --output_dir ${CKPT_ROOT}/msvd \
  --bert_model modules/bert-model --do_lower_case \
  --visual_model modules/visual-base \
  --decoder_model modules/decoder-base \
  --init_model ${INIT_MODEL_PATH} \
  --lr 5e-7 --max_words 48 --max_frames 20 \
  --visual_num_hidden_layers 2 --decoder_num_hidden_layers 2 \
  --video_dim 512 --d_model 512 \
  --datatype msvd

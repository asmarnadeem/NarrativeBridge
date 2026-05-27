#!/bin/bash
# Evaluate a trained CEN checkpoint on the MSRVTT-CTN test split.

set -euo pipefail

N_GPU=1
N_THREAD=8

DATA_PATH=/path/to/MSRVTT_CTN_data.json
CKPT_ROOT=/path/to/checkpoints
INIT_MODEL_PATH=${CKPT_ROOT}/msrvtt/pytorch_model.bin.best
FEATURES_PATH_CAUSE=/path/to/MSRVTT_cause_features.pickle
FEATURES_PATH_EFFECT=/path/to/MSRVTT_effect_features.pickle

python -m torch.distributed.launch --nproc_per_node=${N_GPU} \
  eval_msrvtt.py \
  --do_eval \
  --num_thread_reader ${N_THREAD} \
  --batch_size 64 --batch_size_val 128 \
  --data_path ${DATA_PATH} \
  --features_path_cause ${FEATURES_PATH_CAUSE} \
  --features_path_effect ${FEATURES_PATH_EFFECT} \
  --output_dir ${CKPT_ROOT}/msrvtt_eval \
  --bert_model modules/bert-model --do_lower_case \
  --visual_model modules/visual-base \
  --decoder_model modules/decoder-base \
  --init_model ${INIT_MODEL_PATH} \
  --max_words 48 --max_frames 20 \
  --visual_num_hidden_layers 2 --decoder_num_hidden_layers 2 \
  --video_dim 512 --d_model 512 \
  --datatype msrvtt

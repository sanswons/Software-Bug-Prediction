#!/bin/bash
DATA_DIR=lang_model/data
OUTDIR=./trained_model
rm -rf $OUTDIR
t2t-trainer \
  --data_dir=lang_model/data \
  --t2t_usr_dir=./lang_model/trainer \
  --problem=$PROBLEM \
  --model=transformer \
  --hparams_set=transformer_lang_gen \
  --output_dir=$OUTDIR --job-dir=$OUTDIR --train_steps=200

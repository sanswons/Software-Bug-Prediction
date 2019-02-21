#!/bin/bash
# same as the above training job ...
OUTDIR=./trained_model 
DATADIR=./lang_model/data
MODEL=transformer

BEAM_SIZE=5
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATADIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=transformer_lang_gen \
  --output_dir=$OUTDIR \
  --t2t_usr_dir=./lang_model/trainer \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --hparams='sampling_method=random' \
  --decode_from_file=input.txt \ 

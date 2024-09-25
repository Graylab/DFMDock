#!/bin/bash

# Set default values
DIR=$(pwd)
TRAIN_SET="dips"
TEST_SET="db5_test"
MODEL="model_0"
NUM_SAMPLES=120
NUM_STEPS=40

# Use passed argument for RUN, default to 0 if not provided
RUN=${1:-0}

# Execute the code
python ../src/inference.py \
  data.ckpt=../checkpoints/${TRAIN_SET}/${MODEL}.ckpt \
  data.dataset=${TEST_SET} \
  data.out_csv_dir=csv_files/ \
  data.out_csv=${TEST_SET}_${MODEL}_${NUM_SAMPLES}_samples_${NUM_STEPS}_steps_${TRAIN_SET}_${RUN}.csv \
  data.num_samples=${NUM_SAMPLES} \
  data.num_steps=${NUM_STEPS} \
  data.out_pdb=True \
  data.out_pdb_dir=${DIR}/pdbs/${TEST_SET}_${MODEL}_${NUM_SAMPLES}_samples_${NUM_STEPS}_steps_${TRAIN_SET}/run${RUN} \
  data.test_all=True \
  data.use_clash_force=False

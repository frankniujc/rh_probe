#!/bin/bash

if [ $1 ]
then
    GPU=$1
else
    GPU=0
fi

if [ $2 ]
then
    MODEL_PATH=$2
else
    MODEL_PATH="configs/full_experiments/rh_probe.json"
fi

CUDA_VISIBLE_DEVICES=$GPU python -i scripts/python_code/resume_experiments.py $MODEL_PATH
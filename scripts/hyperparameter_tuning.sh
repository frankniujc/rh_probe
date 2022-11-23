#!/bin/bash

if [ $1 ]
then
    GPU=$1
else
    GPU=0
fi

CUDA_VISIBLE_DEVICES=$GPU python -u scripts/python_code/optuna_tuning.py
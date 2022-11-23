#!/bin/bash

if [ $1 ]
then
    GPU=$1
else
    GPU=0
fi

if [ $2 ]
then
    CONFIG_PATH=$2
else
    CONFIG_PATH="configs/experiments.json"
fi

CUDA_VISIBLE_DEVICES=$GPU python scripts/python_code/conduct_experiments.py -c $CONFIG_PATH
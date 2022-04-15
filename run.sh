#!/bin/bash

mkdir -p model

python train.py
python infer.py

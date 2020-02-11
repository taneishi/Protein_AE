#!/bin/sh
#PBS -o log/%j.out
#PBS -j oe

lscpu | grep 'Model name\\|^CPU(s)'

if [ $PBS_O_HOST ]; then echo $PBS_O_HOST; fi
if [ $PBS_O_WORKDIR ]; then cd $PBS_O_WORKDIR; mkdir -p log; fi

python autoencoder.py

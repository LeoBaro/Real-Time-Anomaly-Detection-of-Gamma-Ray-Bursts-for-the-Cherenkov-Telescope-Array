#!/bin/bash

datasets=(101 201 301 401 501 601)
models=(m1 m2 m3 m4)
training=(light medium heavy)

for d in ${datasets[@]}
do
    for m in ${models[@]}
    do 
        for t in ${training[@]}
        do 
            printf "$d $m $t\n"
            python train.py -m m1 -di 101 -tt light -of training_output_april_2022 -sa 5 10 20 30 40 10 -e 50 -wb 1
        done
    done
done
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
            printf "\n\n\n$d $m $t\n"
            python train.py -m $m -di $d -tt $t -of training_output_april_2022 -sa 1 5 10 20 30 40 10 -e 50 -wb 1
        done
    done
done
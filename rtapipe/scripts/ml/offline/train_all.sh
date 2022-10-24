#!/bin/bash

dc="/data01/homes/baroncelli/phd/rtapipe/lib/dataset/config/agilehost3-prod5.yml"
datasets=(101 201 301 401 501 601)
models=(lstm-m1 lstm-m2 lstm-m3 lstm-m4)
training=(light medium heavy)
mkdir -p logs
for d in ${datasets[@]}
do
    for m in ${models[@]}
    do 
        for t in ${training[@]}
        do 
            printf "\n\n\n$d $m $t\n"
            #nohup python train.py -m $m -di $d -tt $t -of training_output_june_2022_prod5 -sa 1 5 10 20 30 40 50 100 -e 100 -wb 1 -dc $dc > logs/train-$d-$m-$t.log 2>&1 &
            python train.py -m $m -di $d -tt $t -of training_output_october_2022_prod5 -sa 1 5 10 20 30 40 50 100 -e 100 -wb 1 -dc $dc > logs/train-$d-$m-$t.log
        done
    done
done

# Test: 
# python train.py -m lstm-m4 -di 601 -tt heavy -of training_output_october_2022_prod5 -sa 50 100 150 200 -e 200 -wb 1 -dc /data01/homes/baroncelli/phd/rtapipe/lib/dataset/config/agilehost3-prod5.yml > logs/train_23_10_22.log

#!/bin/bash

dc="/data01/homes/baroncelli/phd/rtapipe/lib/dataset/config/agilehost3-prod5.yml"
datasets=(401 501 601) #(101 201 301 401 501 601)
models=(m3 m4) #(m1 m2 m3 m4)
training=(medium heavy) #(light medium heavy)
mkdir -p logs
for d in ${datasets[@]}
do
    for m in ${models[@]}
    do 
        for t in ${training[@]}
        do 
            printf "\n\n\n$d $m $t\n"
            #nohup python train.py -m $m -di $d -tt $t -of training_output_june_2022_prod5 -sa 1 5 10 20 30 40 50 100 -e 100 -wb 1 -dc $dc > logs/train-$d-$m-$t.log 2>&1 &
            python train.py -m $m -di $d -tt $t -of training_output_june_2022_prod5 -sa 1 5 10 20 30 40 50 100 -e 100 -wb 1 -dc $dc > logs/train-$d-$m-$t.log
        done
    done
done


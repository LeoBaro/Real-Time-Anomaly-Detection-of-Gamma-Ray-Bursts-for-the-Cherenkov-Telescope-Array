#!/bin/bash

datasets=[101,201,301,401,501,601]
models=[m1,m2,m3,m4]


python train.py -m m1 -di 101 -tt light -of training_output_april_2022 -sa 10 20 30 40 -e 40 -wb 1


: '
python train.py -m m1 -di 500 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 500 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 500 -tt heavy -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 600 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 600 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 600 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m2 -di 500 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 500 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 500 -tt heavy -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 600 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 600 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 600 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m3 -di 500 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 500 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 500 -tt heavy -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 600 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 600 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 600 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m4 -di 500 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 500 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 500 -tt heavy -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 600 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 600 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 600 -tt heavy -of training_output_10_epochs -sa 10 -e 10


python train.py -m m1 -di 501 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 501 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 501 -tt heavy -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 601 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 601 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 601 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m2 -di 501 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 501 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 501 -tt heavy -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 601 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 601 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 601 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m3 -di 501 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 501 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 501 -tt heavy -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 601 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 601 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 601 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m4 -di 501 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 501 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 501 -tt heavy -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 601 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 601 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 601 -tt heavy -of training_output_10_epochs -sa 10 -e 10


python train.py -m m1 -di 400 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 400 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 400 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m2 -di 400 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 400 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 400 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m3 -di 400 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 400 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 400 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m4 -di 400 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 400 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 400 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m1 -di 401 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 401 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m1 -di 401 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m2 -di 401 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 401 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m2 -di 401 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m3 -di 401 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 401 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m3 -di 401 -tt heavy -of training_output_10_epochs -sa 10 -e 10

python train.py -m m4 -di 401 -tt light -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 401 -tt medium -of training_output_10_epochs -sa 10 -e 10
python train.py -m m4 -di 401 -tt heavy -of training_output_10_epochs -sa 10 -e 10
'
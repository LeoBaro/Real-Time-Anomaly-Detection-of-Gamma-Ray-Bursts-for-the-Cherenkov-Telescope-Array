#!/bin/bash
: '
python train.py -m m1 -di 500 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 500 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 500 -tt heavy -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 600 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 600 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 600 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m2 -di 500 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 500 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 600 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 500 -tt heavy -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 600 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 600 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m3 -di 500 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 500 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 500 -tt heavy -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 600 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 600 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 600 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m4 -di 500 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 500 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 500 -tt heavy -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 600 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 600 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 600 -tt heavy -of training_output_200_epochs -sa 200 -e 200


python train.py -m m1 -di 501 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 501 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 501 -tt heavy -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 601 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 601 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 601 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m2 -di 501 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 501 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 501 -tt heavy -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 601 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 601 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 601 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m3 -di 501 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 501 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 501 -tt heavy -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 601 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 601 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 601 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m4 -di 501 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 501 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 501 -tt heavy -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 601 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 601 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 601 -tt heavy -of training_output_200_epochs -sa 200 -e 200
'


python train.py -m m1 -di 400 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 400 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 400 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m2 -di 400 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 400 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 400 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m3 -di 400 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 400 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 400 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m4 -di 400 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 400 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 400 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m1 -di 401 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 401 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m1 -di 401 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m2 -di 401 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 401 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m2 -di 401 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m3 -di 401 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 401 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m3 -di 401 -tt heavy -of training_output_200_epochs -sa 200 -e 200

python train.py -m m4 -di 401 -tt light -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 401 -tt medium -of training_output_200_epochs -sa 200 -e 200
python train.py -m m4 -di 401 -tt heavy -of training_output_200_epochs -sa 200 -e 200

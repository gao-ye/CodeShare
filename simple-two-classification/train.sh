#!/bin/bash

GPU=5
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_1 ./dataset/train \
	--test_1 ./dataset/test \
	--imgH 112 \
	--imgW 112 \
	--batchSize 256 \
	--niter 1 \
	--lr 0.1 \
	--cuda \


    
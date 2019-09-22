#!/bin/bash

GPU=0
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_1 ~/workspace/Dataset/DataDB/NIPS2014 \
	--test_1 ~/workspace/Dataset/DataDB/IIIT5K_testLmdb \
	--test_2 ~/workspace/Dataset/DataDB/svt_testLmdb \
	--imgH 128 \
	--imgW 400 \

    
#!/usr/bin/env bash

# train
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=4 \
#train.py --model fcn32s --backbone vgg16 --dataset mask \
# --lr 0.01 --epochs 20

export NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
train.py --model deeplabv3 \
    --backbone resnet18 --dataset mask \
    --lr 0.01 --epochs 90 --batch-size 16 --base-size 480 --crop-size 320
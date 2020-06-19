#!/usr/bin/env bash

# train
#export NGPUS=4
#python -m torch.distributed.launch --nproc_per_node=4 \
#train.py --model fcn32s --backbone vgg16 --dataset mask \
# --lr 0.01 --epochs 20

# batch_size是单卡的batch_size
# 4*16->0.01
# 2*16->0.005
# 2*8->0.0025
export NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
train.py --model deeplabv3 \
    --backbone resnet18 --dataset mask \
    --lr 0.01 --epochs 60 --batch-size 32 --base-size 480 --crop-size 320

#export NGPUS=2
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
#train.py --model psanet \
#    --backbone resnet18 --dataset mask \
#    --lr 0.01 --epochs 60 --batch-size 32 --base-size 500 --crop-size 320
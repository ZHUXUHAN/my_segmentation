# for example, evaluate fcn32_vgg16_pascal_voc with 4 GPUs:

python -m torch.distributed.launch --nproc_per_node=2 eval.py --model deeplabv3 \
--backbone resnet18 --dataset mask --iteration 60
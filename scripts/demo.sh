CUDA_VISIBLE_DEVICES=0,1,2,3 python demo.py --model deeplabv3_resnet18_mask \
--input-pic /home/awesome-semantic-segmentation-pytorch/0023c3c5-c0fd-4ab2-85e0-dd00fa2b5f84_1581153710699_align_0.jpg \
--local_rank 0 \
--outdir ./img_align_r18 \
--crop_size 320
CUDA_VISIBLE_DEVICES=0,1,2,3 python demo.py --model deeplabv3_resnet18_mask \
--inputdir /train/trainset/1/img_align \
--local_rank 0 \
--outdir ./img_align_out \
--crop_size 320 \
--iteration 60

python merge_img.py
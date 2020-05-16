import os
import sys
import argparse
import torch
import cv2

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

plt.switch_backend('agg')
import numpy as np
from core.utils.visualize import get_color_pallete
from core.models import get_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='mask', choices=['pascal_voc/pascal_aug/ade20k/citys/mask'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='/home/awesome-semantic-segmentation-pytorch/0025def6-2daa-4334-8cc0-577c3fef286a_1580717471699_align_0.jpg',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./mask2', type=str,
                    help='path to save the predict result')
parser.add_argument('--local_rank', default=0, type=int,
                    help='local_rank')
parser.add_argument('--crop_size', default=320, type=int,
                    help='local_rank')
args = parser.parse_args()


def create_pascal_label_colormap():
    """
    PASCAL VOC 分割数据集的类别标签颜色映射label colormap
    返回:
        可视化分割结果的颜色映射Colormap
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """
    添加颜色到图片，根据数据集标签的颜色映射 label colormap

    参数:
        label: 整数类型的 2D 数组array, 保存了分割的类别标签 label

    返回:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def center_crop(img, crop_size=args.crop_size):
    outsize = crop_size
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    return img


VOC21 = {0: [0, 0, 0], 1: [0, 128, 0], 2: [0, 128, 0], 3: [0, 128, 128], 4: [128, 0, 0], 5: [128, 0, 128],
         6: [128, 128, 0], 7: [128, 128, 128], 8: [0, 0, 64], 9: [0, 0, 192], 10: [0, 128, 64],
         11: [0, 128, 192], 12: [128, 0, 64], 13: [128, 0, 192], 14: [128, 128, 64], 15: [128, 128, 192],
         16: [0, 64, 0], 17: [0, 64, 128], 18: [0, 192, 0], 19: [0, 192, 128], 20: [128, 64, 0],
         255: [255, 255, 255]}


def dict2array(colordict):
    keys = colordict.keys()
    colorarray = np.zeros((len(keys), 3))
    for ids, k in enumerate(keys):
        colorarray[ids] = np.asarray(colordict[k])
    colorarray = np.asarray(colorarray, dtype=np.int)

    return colorarray


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        # transforms.Resize([320, 320]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = get_model(args.model, pretrained=True, root=args.save_folder, local_rank=args.local_rank).to(device)
    print('Finished loading model!')

    model.eval()

    # file_dir = '/train/trainset/1/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'
    # file_dirs = os.listdir(file_dir)
    # for d, dir in enumerate(file_dirs):
    #     files = os.listdir(os.path.join(file_dir, dir))
    #     for i, file in enumerate(files):
    #         print("%d haved done" % i)
    #         with torch.no_grad():
    #             path = os.path.join(file_dir, dir, file)
    #             ori_image = cv2.imread(path)
    #             # ori_image = cv2.resize(ori_image, (400, 400), interpolation=cv2.INTER_CUBIC)
    #             ori_image = ori_image.astype(np.float32)
    #             image = Image.open(path).convert('RGB')
    #             images = transform(image).unsqueeze(0).to(device)
    #             output = model(images)
    #
    #         pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
    #
    #         colormap = dict2array(VOC21)
    #         parsing_color = colormap[pred.astype(np.int)]
    #
    #         idx = np.nonzero(pred)
    #         ori_image[idx[0], idx[1], :] *= 1.0 - 0.4
    #         ori_image += 0.4 * parsing_color
    #
    #         ori_image = ori_image.astype(np.uint8)
    #
    #         outname = os.path.basename(path).replace('.jpg', '.png')
    #         if not os.path.exists(os.path.join(config.outdir, dir)):
    #             os.makedirs(os.path.join(config.outdir, dir))
    #         cv2.imwrite(os.path.join(config.outdir, dir, outname), ori_image)

    files = os.listdir('/train/trainset/1/img_align')
    for i, file in enumerate(files):
        print("%d haved done" % i)
        with torch.no_grad():
            # image = Image.open(config.input_pic).convert('RGB')
            path = os.path.join('/train/trainset/1/img_align', file)
            ori_image = cv2.imread(path)
            ori_image = ori_image.astype(np.float32)
            out_size = args.crop_size
            h = ori_image.shape[0]
            w = ori_image.shape[1]
            x1 = int(round((w - out_size) / 2.))
            y1 = int(round((h - out_size) / 2.))
            ori_image = ori_image[x1:x1+out_size, y1:y1+out_size, :]
            image = Image.open(path).convert('RGB')
            image = center_crop(image)
            images = transform(image).unsqueeze(0).to(device)
            output = model(images)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()

        colormap = dict2array(VOC21)
        parsing_color = colormap[pred.astype(np.int)]

        idx = np.nonzero(pred)
        ori_image[idx[0], idx[1], :] *= 1.0 - 0.4
        ori_image += 0.4 * parsing_color

        ori_image = ori_image.astype(np.uint8)

        outname = os.path.basename(path).replace('.jpg', '.png')
        cv2.imwrite(os.path.join(args.outdir, outname), ori_image)


if __name__ == '__main__':
    demo(args)

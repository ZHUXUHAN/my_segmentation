import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')

imgfile = '/home/awesome-semantic-segmentation-pytorch/0023c3c5-c0fd-4ab2-85e0-dd00fa2b5f84_1581153710699_align_0.jpg'
pngfile = '/home/awesome-semantic-segmentation-pytorch/scripts/eval/0023c3c5-c0fd-4ab2-85e0-dd00fa2b5f84_1581153710699_align_0.png'

img = cv2.imread(imgfile, 1)
mask = cv2.imread(pngfile, 0)

binary , contours, hierarchy  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

img = img[:, :, ::-1]
img[..., 2] = np.where(mask == 1, 255, img[..., 2])

plt.imshow(img)
plt.savefig('./eval/mask.png')

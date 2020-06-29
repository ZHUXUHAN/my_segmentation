# -*- coding: utf-8 -*-
import json
import cv2
import numpy as np
import os
import shutil
import argparse

if __name__ == '__main__':
    save_dir = '/train/trainset/1/Semantic/data/labels'
    save_img = '/train/trainset/1/Semantic/data/img'
    json_path = '/train/trainset/1/face_mask_data/mask_test_20200521_135556.json'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(save_img):
        os.makedirs(save_img)

    file_dir = '/train/trainset/1/'
    files = os.listdir(file_dir)

    with open(json_path) as f:
        json_s = f.readlines()

    # draw roi
    print(len(json_s))
    with open(os.path.join(file_dir, 'test_list.txt'), 'w') as f:
        for i, item_s in enumerate(json_s):

            item_dict = json.loads(item_s)
            result = item_dict['result']
            url_image = item_dict['url_image']
            img_path = os.path.join(file_dir, url_image[8:])
            print(img_path)
            img = cv2.imread(img_path)
            img_h = img.shape[0]
            img_w = img.shape[1]
            img = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)

            save_path = os.path.join(save_dir, url_image[20:]).replace('.jpg', '.png')

            shutil.copy(img_path, os.path.join(save_img, url_image[20:]))

            line = os.path.basename(os.path.join(save_img, url_image[20:]))
            # f.write(line + '\n') # 训练集写这里！

            # for poly in result:
            if len(result) == 0:
                print("0id", i)
                img = cv2.fillPoly(img, [np.array(points, dtype=int)], 0)
                target = np.array(img).astype(np.uint8)
                cv2.imwrite(save_path, img)
            else:
                f.write(line + '\n') # 测试集写这里！
                points = np.array(result[0]['data']).reshape(-1, 2)
                img = cv2.fillPoly(img, [np.array(points, dtype=int)], 1)
                target = np.array(img).astype(np.uint8)
                if np.max(target) != 1:
                    print(np.min(target), np.max(target))
                cv2.imwrite(save_path, img)

            if i % 100==0:
                print('%d have saved done' % (i))
            if i == len(json_s)-1:
                print('%d have saved done, all have saved done' % (i))


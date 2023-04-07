import os
import pandas as pd
import random
import re
import numpy as np
from PIL import Image, ImageEnhance


def rotate(img_path, degree, root, num, img_num):
    img = Image.open(img_path)
    save_path = os.path.join(root, "rotate")
    mkdir(save_path)

    i = 1
    while i <= num:
        img = img.rotate(degree).convert("RGB")
        img_name = os.path.join(root, "rotate", "clips-" + str(img_num) + '.png')
        img.save(img_name, quality=95)
        i += 1
        print('save to ' + img_name)


def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    root = "/home/jht/PycharmProjects/middle"
    csv_root = "/home/jht/PycharmProjects/middle/data/train.csv"
    csv = pd.read_csv(csv_root)
    lists = os.listdir(os.path.join(root, "newpics"))
    ans_list = []
    for img in lists:
        img_num = re.split("[.-]", img)[1]
        generate = random.randint(1, 3)
        if 5001 <= int(img_num) <= 25000:
            if generate == 1:
                degree = 90
            elif generate == 2:
                degree = 180
            else:
                degree = 270
            img_path = os.path.join(root, "newpics", img)
            ans_list.append(csv['clip_count'][int(img_num) - 5001])
            # print(csv['clip_count'][int(img_num) - 5001])
            rotate(img_path, degree, root=root, num=1, img_num=int(img_num))
    print(ans_list)
    pd_target = pd.DataFrame(ans_list, index=range(len(ans_list)), columns=['nums'])
    pd_target.to_csv(os.path.join(root, "rotate", "ans.csv"))

import os.path

from PIL import Image, ImageChops
import numpy as np


def less(t1, t2):
    for i in range(len(t1)):
        if t1[i] > t2[i]:
            return False
    return True


def rm_bk(root, img_name):
    # 不带回形针的路径
    background_path = "/home/jht/PycharmProjects/middle/background/example.png"
    img1_path = os.path.join(root, img_name)
    # "D:\\2022机器学习导论作业数据集\\回形针计数\\datas\\images\\clips-1.png"
    bk_img = Image.open(background_path).convert("RGB")
    exp_img = Image.open(img1_path).convert("RGB")
    ans_img = ImageChops.subtract(bk_img, exp_img, scale=0.5, offset=0)
    # ans1_img = ImageChops.difference(bk_img, exp_img)
    # ans1_array = np.array(ans1_img)
    # ans1_img.show()
    # for i in np.array(ans1_img):
    #     for j in i:
    #         print(j)
    color_0 = ans_img.getpixel((0, 0))
    color_black = (0, 0, 0)
    color_while = (255, 255, 255)
    color_mid = (60, 60, 60)
    color_o = (20, 20, 20)
    # emmm 增强一下
    H, L = ans_img.size
    for i in range(H):
        for j in range(L):
            dot = (i, j)
            color_1 = ans_img.getpixel(dot)
            # 去除红色
            if color_1[0] >= 100 and color_1[1] <= 30 and color_1[2] <= 30:
                ans_img.putpixel(dot, color_while)
            # 去除蓝色
            elif color_1[2] >= 100:
                ans_img.putpixel(dot, color_while)
            # print(color_1)
            if less(color_mid, color_1):
                ans_img.putpixel(dot, color_while)
            # elif less(color_o, color_1):
            #     ans_img.putpixel(dot, color_while)
    path = "/home/jht/PycharmProjects/middle/newpics"
    path = path + "/" + img_name
    ans_img.save(path)


if __name__ == '__main__':
    root = "/home/jht/PycharmProjects/middle/pics"
    for items in os.listdir(root):
        img_name = items
        rm_bk(root, items)

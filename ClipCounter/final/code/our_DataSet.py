import csv
import numpy as np
import torch
import cv2
import torchvision.transforms

from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, csv_file, train, r=False, transform=None):
        self.train = train
        self.r = r
        self.data_list = []
        self.label_list = []
        self.transform = transform
        self.pic_path = "../newpics/clips-"
        if r:
            self.pic_path = "../rotate/clips-"
        self.toTensor = torchvision.transforms.ToTensor()
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                self.data_list.append(row[0])  # data_list仅仅需要拿到字符串信息，用于之后找图片
                if train:  # 只有训练集才有label，测试集没有
                    self.label_list.append([float(row[1])])

        self.len = len(self.data_list)

        # label_list转为ndarry
        self.label_list = np.array(self.label_list)

        print("data_list len:" + str(len(self.data_list)))
        print(f"label_list shape:{self.label_list.shape}")
        print("dataset init end")
        # import ipdb;    ipdb.set_trace()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pic = cv2.imread(self.pic_path + self.data_list[idx] + ".png")
        pic = cv2.resize(pic, (128, 128))
        x = self.toTensor(pic)

        if self.train:
            y = torch.Tensor(self.label_list[idx])
        else:
            y = torch.Tensor([0])
        return x, y


if __name__ == '__main__':
    myDataset = MyDataSet("../data/train.csv", train=True)
    myDataset1 = MyDataSet("../data/test.csv", train=False)

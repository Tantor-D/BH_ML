import csv
import numpy as np
import torch

from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, csv_file, train, transform=None):
        self.train = train
        self.data_list = []
        self.label_list = []
        self.transform = transform
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                this_row = []
                for x in row:
                    this_row.append(float(x))
                if train:  # 训练集的话，最后一项是label，测试集则没有，不用管
                    self.label_list.append([this_row.pop()])
                self.data_list.append(this_row)

        self.len = len(self.data_list)

        # 转为ndarry
        self.data_list = np.array(self.data_list)
        self.label_list = np.array(self.label_list)

        print(self.data_list.shape)
        print(self.label_list.shape)

        # import ipdb;    ipdb.set_trace()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.Tensor(self.data_list[idx])

        if self.train:
            y = torch.Tensor(self.label_list[idx])
        else:
            y = torch.Tensor([0])
        return x, y


if __name__ == '__main__':
    myDataset = MyDataSet("../data/trainDataSet.csv", train=True)
    myDataset1 = MyDataSet("../data/testDataSet.csv", train=False)

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn

from our_DataSet import MyDataSet
import our_model

if __name__ == '__main__':
    # 超参数设置
    epoch = int(100)  # 总训练的轮数
    lr = 1e-5
    batch_size = 32

    # 准备dataloader
    train_data = MyDataSet(csv_file='../data/train.csv', train=True)
    rotate_data = MyDataSet(csv_file='../data/train.csv', train=True, r=True)
    test_data = MyDataSet(csv_file='../data/test.csv', train=False)
    train_data = torch.utils.data.ConcatDataset([train_data, rotate_data])

    # 使用dataLoader来加载数据
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=1)

    # length 长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("train dataSet size: {}".format(train_data_size))
    print("test dataSet size:  {}".format(test_data_size))

    # 模型、优化器、损失函数、device的设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f'../models/model_epoch{1}.pth').to(device)
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 其它
    total_train_step = 0  # 训练的次数
    writer = SummaryWriter("../logs_train")  # 当前文件在src目录下，建一个与src同层的文件夹

    # 开始训练
    for i in range(epoch):  # 一共进行epoch轮训练，包含训练和测试两个小步骤
        print(f'---------No. {i} epoch start----------')

        # 训练步骤开始
        model.train()  # 设置网络为训练状态，需注意的话其仅对部分层有作用，如：dropout，batchnorm
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:  # 每100次输出一次训练的相关信息
                print(f"训练次数：{total_train_step}, Loss: {loss.item()}")
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 每训完一个epoch就存一次
        torch.save(model, f"../models/model_epoch{i}.pth")
        print("model saved")

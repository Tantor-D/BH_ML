import torch
import cv2
import torchvision

epoch_num = int(59) # 可以自己选
device = "cuda:1"   # 这个要自己改，卡0在训练，用卡1做的预测

pic_path = "../pics/clips-"
model = torch.load(f'../models/model_epoch{epoch_num}.pth').to(device)
toTensor = torchvision.transforms.ToTensor()


fp = open('../submission.txt', 'w')
model.eval()
for i in range(1, 5001):
    pic = cv2.imread(pic_path + str(i) + ".png")
    pic = cv2.resize(pic, (128, 128))
    x = toTensor(pic).to(device)
    predict_y = model(x.unsqueeze(0)).item()
    if (i%200 ==0): # 只是为了确定还在运行
        print(f"{i}  {int(round(predict_y))}")
    print(int(round(predict_y)), file=fp)

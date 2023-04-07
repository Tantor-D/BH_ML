import torch
import cv2
import torchvision
from PIL import Image
import torchvision.transforms as transforms

epoch_num = int(58)  # 可以自己选
device = "cuda:1"  # 这个要自己改，卡0在训练，用卡1做的预测

pic_path = "../pics/clips-"
model = torch.load(f'../models-vgg/model_epoch{epoch_num}.pth').to(device)
toTensor = torchvision.transforms.ToTensor()

fp = open('../submission-vgg.txt', 'w')

model.eval()
for i in range(1, 5001):
    pic = Image.open(pic_path + str(i) + ".png")
    pic = pic.convert("RGB")
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transformer(pic).to(device)

    predict_y = model(x.unsqueeze(0)).item()
    if i % 200 == 0:  # 只是为了确定还在运行
        print(f"{i}  {int(round(predict_y))}")
    print(int(round(predict_y, 0)), file=fp)

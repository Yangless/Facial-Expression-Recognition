import cv2
import torch
from torchvision import transforms
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import time
import transforms as transforms
from fer import FER2013

class Res18Feature(nn.Module):
    def __init__(self, pretrained, num_classes = 7):
        super(Res18Feature, self).__init__()
        resnet  = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)

        x = x.view(x.size(0), -1)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out

# 模型存储路径
model_save_path = "ijba.pth.tar"#修改为你自己保存下来的模型文件
#img_path = "test_3066_aligned.jpg"#待测试照片位置
img_path = "test.jpg"#待测试照片位置

# ------------------------ 加载数据 --------------------------- #

preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
#        transforms.Resize((224, 224)),
        transforms.Resize((44,44)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


 
#model.load_state_dict(state_dict_new)


        
res18 = Res18Feature(pretrained = True)
#res18 = nn.DataParallel(res18)
print(res18)
#res18.load_state_dict(torch.load(model_save_path, map_location='cpu')['state_dict'])
checkpoint = torch.load(model_save_path,map_location='cpu')


#from collections import OrderedDict
#state_dict_new = OrderedDict()
#for k, v in checkpoint.items():
#    name = k[7:]  # 去掉 `module.`
#    print(k)
##    print(name)
#    state_dict_new[name] = v
#print(checkpoint)
#print(state_dict_new)
#res18.load_state_dict(state_dict_new)

from collections import OrderedDict
state_dict_new = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:]  # 去掉 `module.`
#    name='features.'+name
    print(k)
#    print(name)
    state_dict_new[name] = v


#res18.load_state_dict(checkpoint['state_dict'])
#res18.load_state_dict(state_dict_new)





#res18.cuda()
res18.eval()


#transform_train = transforms.Compose([
#    transforms.RandomCrop(44),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#])
#
#trainset = FER2013(split = 'Training', transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
#
#for epoch in range(1):
#    for batch_idx, (inputs, targets) in enumerate(trainloader):


for i in range(6):
    time1=time.time()
    
    image = cv2.imread(img_path)
    image = image[:, :, ::-1] # BGR to RGB
    image_tensor = preprocess_transform(image)
    #print(image_tensor.shape)
    tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)

#    print(tensor.shape) #[1,3, 224, 224]
#    tensor=tensor.cuda()
    #print(tensor.shape)

    time2=time.time()
    _, outputs = res18(tensor)
    _, predicts = torch.max(outputs, 1)
    time3=time.time()
    print(outputs)
    print(predicts)
#    print((time2-time1)*1000,(time3-time2)*1000,(time3-time1)*1000)

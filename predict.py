# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:31:38 2021

@author: Y
"""

import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import torch
#import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
#from skimage import io
from skimage.transform import resize
from models import *
from models.resnet import ResNet 


#torch.cuda.set_device(-1)
cut_size = 44               #44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


class vgg2pre():
    def __init__(self):
        self.net=VGG()
        self.checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'),map_location='cpu' )    
        self.net.load_state_dict(self.checkpoint['net'])
        self.net.eval()

#class easy2pre():
#    def __init__(self):
#        self.net=Face_CNN()
##        model_parm =  'model_net1.pkl'
#        self.net.load_state_dict(torch.load(os.path.join('model_net','model_net1.pkl'),map_location='cpu') )
#        self.net.eval()
class res2pre():
    def __init__(self):
        self.net=ResNet()
        self.checkpoint = torch.load(os.path.join('FER2013_ResNet18', 'PrivateTest_model.t7'),map_location='cpu' )    
        self.net.load_state_dict(self.checkpoint['net'])
        self.net.eval()
class densenet2pre():
    def __init__(self):
        self.net= Densenet()
        self.checkpoint = torch.load(os.path.join('FER2013_DenseNet121', 'PrivateTest_model.t7'),map_location='cpu' )    
        self.net.load_state_dict(self.checkpoint['net'])
        self.net.eval()
class se_resnet2pre():
    def __init__(self):
        self.net = SeNet()
        self.checkpoint = torch.load(os.path.join('FER2013_SeNet18', 'PrivateTest_model.t7'),map_location='cpu' )    
        self.net.load_state_dict(self.checkpoint['net'])
        self.net.eval()




def predict(image,net):
#    image = io.imread('images/1.jpg')
#    raw_img=image
#    gray=image
    gray = rgb2gray(image)
    gray = resize(gray, (44,44), mode='symmetric').astype(np.uint8)
#    gray = rgb2gray(image).astype(np.uint8)

#    print(gray)
    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    
    img = Image.fromarray(img)
    inputs = transform_test(img)
        #print(inputs.size())


    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


    

#    net.eval()
    
    ncrops, c, h, w = np.shape(inputs)
    
    inputs = inputs.view(-1, c, h, w)
    #inputs = inputs.cuda()   
    with torch.no_grad():
        inputs = Variable(inputs)
#    print(inputs)
    outputs = net(inputs)              #10*7
        #print(outputs.size())
        #print(outputs)
        #outputs = outputs.view(ncrops, -1).mean(0)
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops    7
        #print(outputs_avg.size())
#    print(outputs_avg)
    score = F.softmax(outputs_avg,dim=0)
#    print(score.size())
#    print(score) 
    _, predicted = torch.max(score.data, 0)        #0是每列的最大值，1是每行的最大值
    #print(_,predicted)
        
        
#    plt.savefig(os.path.join('images/results/1.png'))
#    plt.close()
        
#    print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
    return str(class_names[int(predicted.cpu().numpy())])
#print(predict(""))
#vgg=vgg2pre()
#print(vgg.net.named_parameters)
#for name, param in vgg.net.named_parameters():
#	print(name, '      ', param)
#image = io.imread('images/2.jpg')
#print(predict(image,vgg.net)) 
#
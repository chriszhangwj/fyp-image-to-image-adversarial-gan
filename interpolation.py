import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from generators import Generator_CIFAR10 as Generator
from noise import pnoise2
from utils import tile_evolution
from math import log10
from prepare_dataset import load_dataset

device = 'cuda'

## ----------------------------------------------------MNIST--------------------------------------------------------
## load generator
#G = Generator()
#checkpoint_name_G = 'Model_C_untargeted.pth.tar'
#checkpoint_path_G = os.path.join('saved', 'baseline', checkpoint_name_G)
#checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
#G.load_state_dict(checkpoint_G['state_dict'])
#G.eval()
#G.cuda()
#
## read images and compute adversaries
#digit_1 = 5 # starting
#digit_2 = 1 # finishing
#img_1_path = 'images/%d.jpg'%(digit_1)
#img_2_path = 'images/%d.jpg'%(digit_2)
#
#img_1 = cv2.imread(img_1_path, cv2.IMREAD_GRAYSCALE)
#img_1 = img_1.copy().astype(np.float32)
#img_1 = img_1[None, None, :, :]/255.0 # normalise the image to [0,1]
#img_1 = torch.from_numpy(img_1) # convert numpy array to a tensor
#img_1 = img_1.cuda()
#img_1_adv = G(img_1).data.clamp(min=0, max=1) # use the pre-trained G to produce perturbation
#
#img_2 = cv2.imread(img_2_path, cv2.IMREAD_GRAYSCALE)
#img_2 = img_2.copy().astype(np.float32)
#img_2 = img_2[None, None, :, :]/255.0 # normalise the image to [0,1]
#img_2 = torch.from_numpy(img_2) # convert numpy array to a tensor
#img_2 = img_2.cuda()
#img_2_adv = G(img_2).data.clamp(min=0, max=1) # use the pre-trained G to produce perturbation
#
## interpolate
#n_intp = 20
#
##for i in range(1, n_intp+1):
##    alpha = float(i)/float(n_intp+1)
##    x_intp = Variable(torch.FloatTensor(img_1.size(2), img_1.size(3)).fill_(0.0).to(device), requires_grad=False)
##    x_intp.data = img_1 * (1.0-alpha) + img_2 * (alpha)
##    img_fake_intp = G(x_intp)

##    
##    # save interpolated input
##    fname_input = os.path.join('images','interpolation','5to1','input_%d.png'%(i))
##    x_intp_img = x_intp.cpu().detach().numpy().squeeze()*255
##    cv2.imwrite(fname_input, x_intp_img)
##    
##    # save interpolated output
##    img_fake_intp = img_fake_intp.cpu().detach().numpy().squeeze()*255
##    fname = os.path.join('images','interpolation','5to1','%d.png'%(i))
##    cv2.imwrite(fname, img_fake_intp)
#    
#def tile_interpolation(num_intp):
#    arr = np.zeros((28*6, 28*num_intp), dtype=np.uint8)
#    for i in range(num_intp):
#        path = 'images/interpolation/0to7/input_%d.png'%(i+1)
#        print(path)
#        img_input = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#        print(img_input)
#        path = 'images/interpolation/0to7/%d.png'%(i+1)
#        img_output = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#        a=0
#        b=i
#        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_input
#        a=1
#        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_output
#        
#    for i in range(num_intp):
#        path = 'images/interpolation/8to4/input_%d.png'%(i+1)
#        print(path)
#        img_input = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#        print(img_input)
#        path = 'images/interpolation/8to4/%d.png'%(i+1)
#        img_output = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#        a=2
#        b=i
#        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_input
#        a=3
#        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_output
#        
#    for i in range(num_intp):
#        path = 'images/interpolation/5to1/input_%d.png'%(i+1)
#        print(path)
#        img_input = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#        print(img_input)
#        path = 'images/interpolation/5to1/%d.png'%(i+1)
#        img_output = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#        a=4
#        b=i
#        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_input
#        a=5
#        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_output                    
#    plt.figure(figsize=(11,11))
#    plt.imshow(arr, cmap = 'gray')
#    plt.axis('off')
#    plt.show()    
#    
## ----------------------------------------------------CIFAR10-------------------------------------------------------
## load generator
#G = Generator()
#checkpoint_name_G = 'ours.pth.tar'
#checkpoint_path_G = os.path.join('saved', 'cifar10','ours', checkpoint_name_G)
#checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
#G.load_state_dict(checkpoint_G['state_dict'])
#G.eval()
#G.cuda()
#
## read images and compute adversaries
## dog 1; cat 2; horse 3; deer 4; plane 5; ship 6; automobile 7; bird 8
#img_1_path = 'images/cifar10/interpolation/1160.png'
#img_2_path = 'images/cifar10/interpolation/417.png'
#
#img_1 = cv2.imread(img_1_path, cv2.IMREAD_COLOR)
#img_1 = img_1.copy().astype(np.float32) # convert double to float [32,32,3] 
#img_1 = np.transpose(img_1,(2,0,1))
#img_1 = img_1/255.0 # [0,1]
#img_1 = img_1*2-1 # [-1,1]
#img_1 = torch.from_numpy(img_1) # convert numpy array to a tensor
#img_1 = img_1.unsqueeze(0)
#img_1 = img_1.cuda()
#
#img_2 = cv2.imread(img_2_path, cv2.IMREAD_COLOR)
#img_2 = img_2.copy().astype(np.float32)
#img_2 = np.transpose(img_2,(2,0,1))
#img_2 = img_2/255.0
#img_2 = img_2*2-1
#img_2 = torch.from_numpy(img_2)
#img_2 = img_2.unsqueeze(0)
#img_2 = img_2.cuda()
#
#n_intp = 20
#for i in range(1, n_intp+1):
#    alpha = float(i)/float(n_intp+1)
#    x_intp = Variable(torch.FloatTensor(img_1.size(2), img_1.size(3)).fill_(0.0).to(device), requires_grad=False)
#    x_intp.data = img_1 * (1.0-alpha) + img_2 * (alpha) # [-1,1]
#    img_fake_intp = G(x_intp)
#    img_fake_intp = img_fake_intp.clamp(min=-1,max=1) # [-1,1]
#    
#    # save interpolated input
#    fname_input = os.path.join('images','cifar10','interpolation','5to5','input_%d.png'%(i))
#    x_intp_img = x_intp.cpu().detach().numpy().squeeze()
#    x_intp_img = np.transpose(x_intp_img,(1,2,0)) # [32*32*3]
#    x_intp_img = (x_intp_img/2+0.5)*255.0
#    cv2.imwrite(fname_input, x_intp_img)
#    
#    # save interpolated output
#    img_fake_intp = img_fake_intp.cpu().detach().numpy().squeeze()
#    fname = os.path.join('images','cifar10','interpolation','5to5','%d.png'%(i))
#    img_fake_intp = np.transpose(img_fake_intp,(1,2,0))
#    img_fake_intp = (img_fake_intp/2+0.5)*255.0
#    img_fake_intp = img_fake_intp.astype(np.float64)
#    #img_fake_intp = 
#    cv2.imwrite(fname, img_fake_intp)    
    
def tile_interpolation_cifar10(num_intp):
    arr = np.zeros((32*6, 32*num_intp,3), dtype=np.float64)
    for i in range(num_intp):
        path = 'images/cifar10/interpolation/1to7/input_%d.png'%(i+1)
        img_input = cv2.imread(path, cv2.IMREAD_COLOR)
        img_input = img_input/255.0
        path = 'images/cifar10/interpolation/1to7/%d.png'%(i+1)
        img_output = cv2.imread(path, cv2.IMREAD_COLOR)
        img_output = img_output/255.0
        a=0
        b=i
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_input
        a=1
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_output
        
    for i in range(num_intp):
        path = 'images/cifar10/interpolation/5to5/input_%d.png'%(i+1)
        img_input = cv2.imread(path, cv2.IMREAD_COLOR)
        img_input = img_input/255.0
        path = 'images/cifar10/interpolation/5to5/%d.png'%(i+1)
        img_output = cv2.imread(path, cv2.IMREAD_COLOR)
        img_output = img_output/255.0
        a=2
        b=i
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_input
        a=3
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_output
        
    for i in range(num_intp):
        path = 'images/cifar10/interpolation/6to3/input_%d.png'%(i+1)
        img_input = cv2.imread(path, cv2.IMREAD_COLOR)
        img_input = img_input/255.0
        path = 'images/cifar10/interpolation/6to3/%d.png'%(i+1)
        img_output = cv2.imread(path, cv2.IMREAD_COLOR)
        img_output = img_output/255.0
        a=4
        b=i
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_input
        a=5
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_output                    
    plt.figure(figsize=(18,18))
    plt.imshow(arr, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()    

tile_interpolation_cifar10(num_intp=20)


#def cifar10_save_png():
#    train_data, test_data, in_channels, num_classes = load_dataset('cifar10')
#    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
#    device='cuda'
#    for i, (img, label) in enumerate(test_loader):
#        if i%100==0:
#            print(i)
#        path ='images/cifar10/cifar10_png/%d.png'%(i)
#        img_real = Variable(img.to(device))
#        img_real = img_real.cpu()
#        img_real = img_real.data.squeeze().numpy() # [nbatch,3,32,32]
#        img_real = img_real/2+0.5 # [0,1]
#        img = img_real*255 # [0,255]
#        img = img.astype(np.float64)
#        img = np.transpose(img, (1, 2, 0)) # [32,32,3]
#        cv2.imwrite(path, img) 
#cifar10_save_png()


## path = 'images/cifar10/interpolation/1to7/input_1.png'
#path = 'images/cifar10/train_evolution/3/3_epoch_0.png'
#img_input = cv2.imread(path, cv2.IMREAD_COLOR)
#plt.figure(figsize=(4,4))
#plt.imshow(img_input)
#plt.axis('off')
#plt.show() 
#
#path = 'images/cifar10/interpolation/1to7/input_1.png'
#img_input = cv2.imread(path, cv2.IMREAD_COLOR)
##img_input = img_input/255.0
##img_input = img_input.astype(np.float64)
##print(img_input)
#plt.figure(figsize=(4,4))
#plt.imshow(img_input)
#plt.axis('off')
#plt.show()    
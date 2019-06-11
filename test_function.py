import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_ssim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from noise import pnoise2
from utils import toZeroThreshold
from math import log10
from skimage.measure import compare_psnr
torch.manual_seed(0)

def test(G, f, target, is_targeted, thres, test_loader, epoch, epochs, device, verbose=True):
    n = 0
    acc = 0
    ssim = 0

    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)

        if is_targeted: # if targeted
            y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else: # if untargeted
            y_true = Variable(label.to(device))
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img.size(0)
        
    img_real = img_real.cpu()
    img_real = img_real.data.squeeze().numpy()
    plt.figure(figsize=(1.5,1.5))
    plt.imshow(img_real[1,:,:], cmap = 'gray')
    plt.title('Real image: digit %d'%(label[1]))
    plt.show()    
    
    
    img_fake = img_fake.cpu()
    adversarial_img = img_fake.data.squeeze().numpy()
    label = label.cpu()
    label = label.data.squeeze().numpy()
    plt.figure(figsize=(1.5,1.5))
    plt.imshow(adversarial_img[1,:,:], cmap = 'gray')
    plt.title('Real image: digit %d'%(label[1]))
    plt.show()    
    
    plt.figure(figsize=(1.5,1.5))
    plt.imshow(img_real[3,:,:], cmap = 'gray')
    plt.title('Real image: digit %d'%(label[1]))
    plt.show()    
    
    plt.figure(figsize=(1.5,1.5))
    plt.imshow(adversarial_img[3,:,:], cmap = 'gray')
    plt.title('Real image: digit %d'%(label[1]))
    plt.show() 

    return acc/n, ssim/n # returns attach success rate

def test_cifar10(G, f, target, is_targeted, thres, test_loader, epoch, epochs, device, verbose=True):
    n = 0
    acc = 0
    ssim = 0
    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))
        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=-1, max=1)
        y_pred = f(img_fake)

        if is_targeted: # if targeted
            y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else: # if untargeted
            y_true = Variable(label.to(device))
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img.size(0)
        
    img_real = img_real.cpu()
    img_real = img_real.data.squeeze().numpy() # [nbatch,3,32,32]
    plt.figure(figsize=(1.5,1.5))
    img_real = img_real[1,:,:,:] # [1,3,32,32]
    img_real = img_real.squeeze()
    img_real = np.transpose(img_real, (1, 2, 0))
    img_real = img_real/2 + 0.5
    plt.imshow(img_real)
    plt.show()    
    
    img_fake = img_fake.cpu()
    img_fake = img_fake.data.squeeze().numpy()
    label = label.cpu()
    label = label.data.squeeze().numpy()
    plt.figure(figsize=(1.5,1.5))
    img_fake = img_fake[1,:,:,:]
    img_fake = img_fake.squeeze()
    img_fake = np.transpose(img_fake, (1, 2, 0))
    img_fake = img_fake/2 + 0.5
    plt.imshow(img_fake)
    plt.show()

    return acc/n, ssim/n # returns attach success rat

#def test_semitargeted(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
#    n = 0
#    acc = 0
#    ssim = 0
#
#    G.eval()
#    for i, (img, label) in enumerate(test_loader):
#        img_real = Variable(img.to(device))
#
#        pert = torch.clamp(G(img_real), -thres, thres)
#        img_fake = pert + img_real
#        img_fake = img_fake.clamp(min=0, max=1)
#
#        y_pred = f(img_fake)
#
#        y_true = Variable(label.to(device))
#        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
#        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
#        n += img.size(0)
##        if verbose:
##            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
#    return acc/n, ssim/n # returns attach success rate
#
#def test_semitargeted_targeted(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
#    n = 0
#    acc = 0
#    ssim = 0
#    target_pair = {0:5,1:8,2:8,3:5,4:9,5:3,6:2,7:9,8:5,9:4} # use class-wise L2 dict to evaluate
#    
#    G.eval()
#    for i, (img, label) in enumerate(test_loader):
#        img_real = Variable(img.to(device))
#
#        pert = torch.clamp(G(img_real), -thres, thres)
#        img_fake = pert + img_real
#        img_fake = img_fake.clamp(min=0, max=1)
#
#        y_pred = f(img_fake)
#         # determine the corresponding target label
#        y_target=[]
#        #print(label)
#        for x in label.numpy():
#            #target_pair.get(x)
#             y_target.append(target_pair.get(x))
#        y_target = torch.LongTensor(y_target).to(device)
#        acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
#        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
#        n += img.size(0)
##        if verbose:
##            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
#    return acc/n, ssim/n # returns attach success rate

def test_baseline(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
    n = 0
    acc = 0
    ssim = 0

    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))

        img_fake = torch.clamp(G(img_real), 0, 1)
        pert = img_fake - img_real

        y_pred = f(img_fake)

        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img.size(0)
      
    img_real = img_real.cpu()
    img_real = img_real.data.squeeze().numpy()
    plt.figure(figsize=(2,2))
    plt.imshow(img_real[1,:,:], cmap = 'gray')
    plt.title('Real image: digit %d'%(label[1]))
    plt.show()    
    
    
    img_fake = img_fake.cpu()
    adversarial_img = img_fake.data.squeeze().numpy()
    label = label.cpu()
    label = label.data.squeeze().numpy()
    plt.figure(figsize=(2,2))
    plt.imshow(adversarial_img[1,:,:], cmap = 'gray')
    plt.title('Real image: digit %d'%(label[1]))
    plt.show()    

#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    return acc/n, ssim/n # returns attach success rate

def test_perlin(G, f, M, test_loader, epoch, epochs, device, verbose=True):
    n, acc, ssim, psnr = 0, 0, 0, 0
    class_acc = np.zeros((1,10)) # count the number of success for each class
    
    noise = perlin(size = 28, period = 60, octave = 1, freq_sine = 36) # [0,1]
    noise = (noise - 0.5)*2 # [-1,1]
    #payload = (np.sign(noise.reshape(28, 28, 1)) + 1) / 2 # [-1,1] binary # [0,2] binary # [0,1] binary
    noise = M * noise.squeeze()

    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img = img.cpu()
        img = img.detach().numpy()
        img_real = 255 * img
        img_noise = np.tile(noise,(img_real.shape[0],1,1,1))
        img_real = img_real + img_noise
        img_real = img_real/255.0
        img_real = Variable(torch.from_numpy(img_real).to(device))
        
        img_fake = torch.clamp(G(img_real), 0, 1)
        #pert = img_fake - img_real

        y_pred = f(img_fake)

        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        #ssim += pytorch_ssim.ssim(img_real, img_fake).item() # need to set batch size = 1
        # compute psnr
#        img_real = img_real.cpu()
#        img_real = img_real.data.numpy()
#        img_fake = img_fake.cpu()
#        img_fake = img_fake.data.numpy()
        #psnr += compare_psnr(img_real,img_fake)
        n += img_real.shape[0] # img_real is numpy array
    #print('ssim:', ssim/n)
    #print('psnr:', psnr/n)  
    print(np.shape(img_real))
    print(type(img_real))
    img_real = img_real.squeeze()
    print(np.shape(img_real))
    path = 'images/train_evolution'
    img_real = img_real.cpu()
    img_real = img_real.data.squeeze().numpy()
    img_fake = img_fake.cpu()
    img_fake = img_fake.data.squeeze().numpy()
    
    if epoch == 0:
        real_img_0 = img_real[9,:,:]*255 # restore to [0,255]
        real_img_0 = real_img_0.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '0', '0_epoch_0.png'), real_img_0)
        real_img_1 = img_real[0,:,:]*255 # restore to [0,255]
        real_img_1 = real_img_1.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '1', '1_epoch_0.png'), real_img_1)
        real_img_2 = img_real[1,:,:]*255 # restore to [0,255]
        real_img_2 = real_img_2.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '2', '2_epoch_0.png'), real_img_2)
        real_img_3 = img_real[2,:,:]*255 # restore to [0,255]
        real_img_3 = real_img_3.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '3', '3_epoch_0.png'), real_img_3)
        real_img_4 = img_real[3,:,:]*255 # restore to [0,255]
        real_img_4 = real_img_4.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '4', '4_epoch_0.png'), real_img_4)
        real_img_5 = img_real[4,:,:]*255 # restore to [0,255]
        real_img_5 = real_img_5.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '5', '5_epoch_0.png'), real_img_5)
        real_img_6 = img_real[5,:,:]*255 # restore to [0,255]
        real_img_6 = real_img_6.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '6', '6_epoch_0.png'), real_img_6)
        real_img_7 = img_real[6,:,:]*255 # restore to [0,255]
        real_img_7 = real_img_7.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '7', '7_epoch_0.png'), real_img_7)
        real_img_8 = img_real[7,:,:]*255 # restore to [0,255]
        real_img_8 = real_img_8.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '8', '8_epoch_0.png'), real_img_8)
        real_img_9 = img_real[8,:,:]*255 # restore to [0,255]
        real_img_9 = real_img_9.astype(np.uint8) # set data type
        cv2.imwrite(os.path.join(path, '9', '9_epoch_0.png'), real_img_9)
        
        
    adv_img_0 = img_fake[9,:,:]*255 # restore to [0,255]
    adv_img_0 = adv_img_0.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'0','0_epoch_%d.png'%(epoch+1)), adv_img_0)
    adv_img_1 = img_fake[0,:,:]*255 # restore to [0,255]
    adv_img_1 = adv_img_1.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'1','1_epoch_%d.png'%(epoch+1)), adv_img_1)
    adv_img_2 = img_fake[1,:,:]*255 # restore to [0,255]
    adv_img_2 = adv_img_2.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'2','2_epoch_%d.png'%(epoch+1)), adv_img_2)    
    adv_img_3 = img_fake[2,:,:]*255 # restore to [0,255]
    adv_img_3 = adv_img_3.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'3','3_epoch_%d.png'%(epoch+1)), adv_img_3)
    adv_img_4 = img_fake[3,:,:]*255 # restore to [0,255]
    adv_img_4 = adv_img_4.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'4','4_epoch_%d.png'%(epoch+1)), adv_img_4)
    adv_img_5 = img_fake[4,:,:]*255 # restore to [0,255]
    adv_img_5 = adv_img_5.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'5','5_epoch_%d.png'%(epoch+1)), adv_img_5)    
    adv_img_6 = img_fake[5,:,:]*255 # restore to [0,255]
    adv_img_6 = adv_img_6.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'6','6_epoch_%d.png'%(epoch+1)), adv_img_6)    
    adv_img_7 = img_fake[6,:,:]*255 # restore to [0,255]
    adv_img_7 = adv_img_7.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'7','7_epoch_%d.png'%(epoch+1)), adv_img_7)
    adv_img_8 = img_fake[7,:,:]*255 # restore to [0,255]
    adv_img_8 = adv_img_8.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'8','8_epoch_%d.png'%(epoch+1)), adv_img_8)  
    adv_img_9 = img_fake[8,:,:]*255 # restore to [0,255]
    adv_img_9 = adv_img_9.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'9','9_epoch_%d.png'%(epoch+1)), adv_img_9)  
    
    if epoch == epochs-1: # save final perturbation
        img_pert = img_real - img_fake # range [-1,1]
        img_pert_0 = img_pert[9,:,:]*255 # restore to [0,255]
        img_pert_0 = img_pert_0.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','0_pert.png'), img_pert_0)
            
        img_pert_1 = img_pert[0,:,:]*255 # restore to [0,255]
        img_pert_1 = img_pert_1.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','1_pert.png'), img_pert_1)
        
        img_pert_2 = img_pert[1,:,:]*255 # restore to [0,255]
        img_pert_2 = img_pert_2.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','2_pert.png'), img_pert_2)
        
        img_pert_3 = img_pert[2,:,:]*255 # restore to [0,255]
        img_pert_3 = img_pert_3.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','3_pert.png'), img_pert_3)
        
        img_pert_4 = img_pert[3,:,:]*255 # restore to [0,255]
        img_pert_4 = img_pert_4.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','4_pert.png'), img_pert_4)
        
        img_pert_5 = img_pert[4,:,:]*255 # restore to [0,255]
        img_pert_5 = img_pert_5.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','5_pert.png'), img_pert_5)
        
        img_pert_6 = img_pert[5,:,:]*255 # restore to [0,255]
        img_pert_6 = img_pert_6.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','6_pert.png'), img_pert_6)
        
        img_pert_7 = img_pert[6,:,:]*255 # restore to [0,255]
        img_pert_7 = img_pert_7.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','7_pert.png'), img_pert_7)
        
        img_pert_8 = img_pert[7,:,:]*255 # restore to [0,255]
        img_pert_8 = img_pert_8.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','8_pert.png'), img_pert_8)
        
        img_pert_9 = img_pert[8,:,:]*255 # restore to [0,255]
        img_pert_9 = img_pert_9.astype(np.int16) # set data type
        cv2.imwrite(os.path.join(path,'pert','9_pert.png'), img_pert_9)
    
    #        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    return acc/n, ssim/n # returns attach success rate

def test_baseline_CIFAR10(G, f, test_loader, epoch, epochs, device, verbose=True):
    n, acc, ssim, psnr = 0, 0, 0, 0
    class_acc = np.zeros((1,10)) # count the number of success for each class
    

    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))
        img_fake = torch.clamp(G(img_real), -1, 1)
        y_pred = f(img_fake)
        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        #ssim += pytorch_ssim.ssim(img_real, img_fake).item() # need to set batch size = 1
        # compute psnr
#        img_real = img_real.cpu()
#        img_real = img_real.data.numpy()
#        img_fake = img_fake.cpu()
#        img_fake = img_fake.data.numpy()
        #psnr += compare_psnr(img_real,img_fake)
        n += img_real.shape[0] # img_real is numpy array
    img_real = img_real.cpu()
    img_real = img_real.data.squeeze().numpy() # [nbatch,3,32,32]
    img_real = img_real/2+0.5
    plt.figure(figsize=(1.5,1.5))
    real_img = img_real[1,:,:,:] # [1,3,32,32]
    real_img = real_img.squeeze()
    real_img = np.transpose(real_img, (1, 2, 0))
    plt.imshow(real_img)
    plt.show()    
    
    img_fake = img_fake.cpu()
    img_fake = img_fake.data.squeeze().numpy()
    img_fake = img_fake/2 + 0.5
    label = label.cpu()
    label = label.data.squeeze().numpy()
    plt.figure(figsize=(1.5,1.5))
    fake_img = img_fake[1,:,:,:]
    fake_img = fake_img.squeeze()
    fake_img = np.transpose(fake_img, (1, 2, 0))
    plt.imshow(fake_img)
    plt.show()
    path = 'images/cifar10/train_evolution'
    if epoch == 0:
        real_img_0 = img_real[9,:,:,:]*255 # restore to [0,255]
        real_img_0 = real_img_0.astype(np.float64) # set data type
        real_img_0 = np.transpose(real_img_0, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '0', '0_epoch_0.png'), real_img_0)
        real_img_1 = img_real[0,:,:,:]*255 # restore to [0,255]
        real_img_1 = real_img_1.astype(np.float64) # set data type
        real_img_1 = np.transpose(real_img_1, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '1', '1_epoch_0.png'), real_img_1)
        real_img_2 = img_real[1,:,:,:]*255 # restore to [0,255]
        real_img_2 = real_img_2.astype(np.float64) # set data type
        real_img_2 = np.transpose(real_img_2, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '2', '2_epoch_0.png'), real_img_2)
        real_img_3 = img_real[2,:,:,:]*255 # restore to [0,255]
        real_img_3 = real_img_3.astype(np.float64) # set data type
        real_img_3 = np.transpose(real_img_3, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '3', '3_epoch_0.png'), real_img_3)
        real_img_4 = img_real[3,:,:,:]*255 # restore to [0,255]
        real_img_4 = real_img_4.astype(np.float64) # set data type
        real_img_4 = np.transpose(real_img_4, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '4', '4_epoch_0.png'), real_img_4)
        real_img_5 = img_real[4,:,:,:]*255 # restore to [0,255]
        real_img_5 = real_img_5.astype(np.float64) # set data type
        real_img_5 = np.transpose(real_img_5, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '5', '5_epoch_0.png'), real_img_5)
        real_img_6 = img_real[5,:,:,:]*255 # restore to [0,255]
        real_img_6 = real_img_6.astype(np.float64) # set data type
        real_img_6 = np.transpose(real_img_6, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '6', '6_epoch_0.png'), real_img_6)
        real_img_7 = img_real[6,:,:,:]*255 # restore to [0,255]
        real_img_7 = real_img_7.astype(np.float64) # set data type
        real_img_7 = np.transpose(real_img_7, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '7', '7_epoch_0.png'), real_img_7)
        real_img_8 = img_real[7,:,:,:]*255 # restore to [0,255]
        real_img_8 = real_img_8.astype(np.float64) # set data type
        real_img_8 = np.transpose(real_img_8, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '8', '8_epoch_0.png'), real_img_8)
        real_img_9 = img_real[8,:,:,:]*255 # restore to [0,255]
        real_img_9 = real_img_9.astype(np.float64) # set data type
        real_img_9 = np.transpose(real_img_9, (1, 2, 0))
        cv2.imwrite(os.path.join(path, '9', '9_epoch_0.png'), real_img_9)
        
    adv_img_0 = img_fake[9,:,:,:]*255 # restore to [0,255]
    adv_img_0 = adv_img_0.astype(np.float64) # set data type
    adv_img_0 = np.transpose(adv_img_0, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'0','0_epoch_%d.png'%(epoch+1)), adv_img_0)
    adv_img_1 = img_fake[0,:,:,:]*255 # restore to [0,255]
    adv_img_1 = adv_img_1.astype(np.float64) # set data type
    adv_img_1 = np.transpose(adv_img_1, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'1','1_epoch_%d.png'%(epoch+1)), adv_img_1)
    adv_img_2 = img_fake[1,:,:,:]*255 # restore to [0,255]
    adv_img_2 = adv_img_2.astype(np.float64) # set data type
    adv_img_2 = np.transpose(adv_img_2, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'2','2_epoch_%d.png'%(epoch+1)), adv_img_2)    
    adv_img_3 = img_fake[2,:,:,:]*255 # restore to [0,255]
    adv_img_3 = adv_img_3.astype(np.float64) # set data type
    adv_img_3 = np.transpose(adv_img_3, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'3','3_epoch_%d.png'%(epoch+1)), adv_img_3)
    adv_img_4 = img_fake[3,:,:,:]*255 # restore to [0,255]
    adv_img_4 = adv_img_4.astype(np.float64) # set data type
    adv_img_4 = np.transpose(adv_img_4, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'4','4_epoch_%d.png'%(epoch+1)), adv_img_4)
    adv_img_5 = img_fake[4,:,:,:]*255 # restore to [0,255]
    adv_img_5 = adv_img_5.astype(np.float64) # set data type
    adv_img_5 = np.transpose(adv_img_5, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'5','5_epoch_%d.png'%(epoch+1)), adv_img_5)    
    adv_img_6 = img_fake[5,:,:,:]*255 # restore to [0,255]
    adv_img_6 = adv_img_6.astype(np.float64) # set data type
    adv_img_6 = np.transpose(adv_img_6, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'6','6_epoch_%d.png'%(epoch+1)), adv_img_6)    
    adv_img_7 = img_fake[6,:,:,:]*255 # restore to [0,255]
    adv_img_7 = adv_img_7.astype(np.float64) # set data type
    adv_img_7 = np.transpose(adv_img_7, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'7','7_epoch_%d.png'%(epoch+1)), adv_img_7)
    adv_img_8 = img_fake[7,:,:,:]*255 # restore to [0,255]
    adv_img_8 = adv_img_8.astype(np.float64) # set data type
    adv_img_8 = np.transpose(adv_img_8, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'8','8_epoch_%d.png'%(epoch+1)), adv_img_8)  
    adv_img_9 = img_fake[8,:,:,:]*255 # restore to [0,255]
    adv_img_9 = adv_img_9.astype(np.float64) # set data type
    adv_img_9 = np.transpose(adv_img_9, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'9','9_epoch_%d.png'%(epoch+1)), adv_img_9)  
    return acc/n, ssim/n # returns attach success rate


def eval_baseline(G, f, M, test_loader, epoch, epochs, device, verbose=True):
    n = 0 # count total number of samples
    n_success = 0 # count total number of successful samples
    acc = 0
    ssim = 0
    psnr = 0 # for PSNR
    distort_success = 0 # for average distortion for successful images
    distort_all = 0 # for average distortion for all images
    
    criterionMSE = nn.MSELoss().to(device) # for PSNR
    class_acc = np.zeros((1,10))
    class_num = np.zeros((1,10))
    noise = perlin(size = 28, period = 60, octave = 1, freq_sine = 36) # [0,1]
    noise = (noise - 0.5)*2 # [-1,1]
    #payload = (np.sign(noise.reshape(28, 28, 1)) + 1) / 2 # [-1,1] binary # [0,2] binary # [0,1] binary
    noise = M * noise.squeeze()
    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img = img.cpu()
        img = img.detach().numpy()
        img_real = 255 * img
        img_noise = np.tile(noise,(img_real.shape[0],1,1,1))
        img_real = img_real + img_noise
        img_real = img_real/255.0
        img_real = Variable(torch.from_numpy(img_real).to(device))
        
        img_fake = torch.clamp(G(img_real), 0, 1)

        y_pred = f(img_fake)
        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        mse = criterionMSE(y_pred,y_true.float()) # for PSNR
        pnsr_temp = 10 * log10(1/mse.item())
        psnr += pnsr_temp
        distort_all_temp = torch.dist(img_real,img_fake, 1) # use L1 loss
        distort_all += distort_all_temp
        class_num[0,y_true] = class_num[0,y_true]+1
        if torch.max(y_pred, 1)[1] != y_true:
            class_acc[0,y_true] = class_acc[0,y_true]+1
            distort_success_temp = torch.dist(img_real, img_fake, 1) # use L1 loss
            distort_success += distort_success_temp
            n_success += 1
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img_real.size(0)
        if i % 100 == 0:
            print(i)
#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    # count number of samples for each class
    
    acc_class = np.divide(class_acc,class_num)
    n_pixel = img_real.size(2) * img_real.size(2)
    return acc/n, ssim/n, psnr/n, acc_class, distort_success/(n_success*n_pixel), distort_all/(n*n_pixel) # returns attach success rateclass_accclass_acc

def eval_advgan(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
    n = 0 # count total number of samples
    n_success = 0 # count total number of successful samples
    acc = 0
    ssim = 0
    psnr = 0 # for average PSNR
    distort_success = 0 # for average distortion for successful images
    distort_all = 0 # for average distortion for all images
    
    criterionMSE = nn.MSELoss().to(device) # for PSNR
    class_acc = np.zeros((1,10))
    class_num = np.zeros((1,10))
    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)
        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        mse = criterionMSE(y_pred,y_true.float()) # for PSNR
        pnsr_temp = 10 * log10(1/mse.item())
        psnr += pnsr_temp
        distort_all_temp = torch.dist(img_real, img_fake, 1) # use L1 loss
        distort_all += distort_all_temp
        class_num[0,y_true] = class_num[0,y_true]+1
        if torch.max(y_pred, 1)[1] != y_true:
            class_acc[0,y_true] = class_acc[0,y_true]+1
            distort_success_temp = torch.dist(img_real, img_fake, 1) # use L1 loss
            distort_success += distort_success_temp
            n_success += 1
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img_real.size(0)
        if i % 100 == 0:
            print(i)
    acc_class = np.divide(class_acc,class_num)
    n_pixel = img_real.size(2) * img_real.size(2)
    
    return acc/n, ssim/n, psnr/n, acc_class, distort_success/(n_success*n_pixel), distort_all/(n*n_pixel) # returns attach success rateclass_accclass_acc

def eval_advgan_batch(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
    acc = 0
    n = 0
    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)
        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        n += img_real.size(0)
        print(i)

    path = 'images/advgan'
    # save real images
    img_real = img_real.cpu()
    img_real = img_real.data.squeeze().numpy()
    real_img_0 = img_real[9,:,:]*255 # restore to [0,255]
    real_img_0 = real_img_0.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '0.png'), real_img_0)
    real_img_1 = img_real[0,:,:]*255 # restore to [0,255]
    real_img_1 = real_img_1.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '1.png'), real_img_1)
    real_img_2 = img_real[1,:,:]*255 # restore to [0,255]
    real_img_2 = real_img_2.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '2.png'), real_img_2)
    real_img_3 = img_real[2,:,:]*255 # restore to [0,255]
    real_img_3 = real_img_3.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '3.png'), real_img_3)
    real_img_4 = img_real[3,:,:]*255 # restore to [0,255]
    real_img_4 = real_img_4.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '4.png'), real_img_4)
    real_img_5 = img_real[4,:,:]*255 # restore to [0,255]
    real_img_5 = real_img_5.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '5.png'), real_img_5)
    real_img_6 = img_real[5,:,:]*255 # restore to [0,255]
    real_img_6 = real_img_6.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '6.png'), real_img_6)
    real_img_7 = img_real[6,:,:]*255 # restore to [0,255]
    real_img_7 = real_img_7.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '7.png'), real_img_7)
    real_img_8 = img_real[7,:,:]*255 # restore to [0,255]
    real_img_8 = real_img_8.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '8.png'), real_img_8)
    real_img_9 = img_real[8,:,:]*255 # restore to [0,255]
    real_img_9 = real_img_9.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path, '9.png'), real_img_9)
    # save fake images
    img_fake = img_fake.cpu()
    img_fake = img_fake.data.squeeze().numpy()
    adv_img_0 = img_fake[9,:,:]*255 # restore to [0,255]
    adv_img_0 = adv_img_0.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'0_adv.png'), adv_img_0)
    adv_img_1 = img_fake[0,:,:]*255 # restore to [0,255]
    adv_img_1 = adv_img_1.astype(np.uint8) # set data type
    cv2.imwrite(os.path.join(path,'1_adv.png'), adv_img_1)
    adv_img_2 = img_fake[1,:,:]*255 # restore to [0,255]
    adv_img_2 = adv_img_2.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'2_adv.png'), adv_img_2)   
    adv_img_3 = img_fake[2,:,:]*255 # restore to [0,255]
    adv_img_3 = adv_img_3.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'3_adv.png'), adv_img_3)
    adv_img_4 = img_fake[3,:,:]*255 # restore to [0,255]
    adv_img_4 = adv_img_4.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'4_adv.png'), adv_img_4)
    adv_img_5 = img_fake[4,:,:]*255 # restore to [0,255]
    adv_img_5 = adv_img_5.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'5_adv.png'), adv_img_5)   
    adv_img_6 = img_fake[5,:,:]*255 # restore to [0,255]
    adv_img_6 = adv_img_6.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'6_adv.png'), adv_img_6)   
    adv_img_7 = img_fake[6,:,:]*255 # restore to [0,255]
    adv_img_7 = adv_img_7.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'7_adv.png'), adv_img_7)
    adv_img_8 = img_fake[7,:,:]*255 # restore to [0,255]
    adv_img_8 = adv_img_8.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'8_adv.png'), adv_img_8) 
    adv_img_9 = img_fake[8,:,:]*255 # restore to [0,255]
    adv_img_9 = adv_img_9.astype(np.int16) # set data type
    cv2.imwrite(os.path.join(path,'9_adv.png'), adv_img_9) 
    return acc/n

def eval_advgan_batch_cifar10(G, f, thres, test_loader, device, verbose=True):
    acc = 0
    n = 0
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))
        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=-1, max=1)
        y_pred = f(img_fake)
        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        n += img.size(0)
        print(i)

    path = 'images/cifar10/advgan'
    # save real images
    img_real = img_real/2 + 0.5
    img_real = img_real.cpu()
    img_real = img_real.data.squeeze().numpy()
    real_img_0 = img_real[9,:,:,:]*255 # restore to [0,255]
    real_img_0 = real_img_0.astype(np.int16) # set data type
    real_img_0 = np.transpose(real_img_0, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '0.png'), real_img_0)
    real_img_1 = img_real[0,:,:,:]*255 # restore to [0,255]
    real_img_1 = real_img_1.astype(np.int16) # set data type
    real_img_1 = np.transpose(real_img_1, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '1.png'), real_img_1)
    real_img_2 = img_real[1,:,:,:]*255 # restore to [0,255]
    real_img_2 = real_img_2.astype(np.int16) # set data type
    real_img_2 = np.transpose(real_img_2, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '2.png'), real_img_2)
    real_img_3 = img_real[2,:,:,:]*255 # restore to [0,255]
    real_img_3 = real_img_3.astype(np.int16) # set data type
    real_img_3 = np.transpose(real_img_3, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '3.png'), real_img_3)
    real_img_4 = img_real[3,:,:,:]*255 # restore to [0,255]
    real_img_4 = real_img_4.astype(np.int16) # set data type
    real_img_4 = np.transpose(real_img_4, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '4.png'), real_img_4)
    real_img_5 = img_real[4,:,:,:]*255 # restore to [0,255]
    real_img_5 = real_img_5.astype(np.int16) # set data type
    real_img_5 = np.transpose(real_img_5, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '5.png'), real_img_5)
    real_img_6 = img_real[5,:,:,:]*255 # restore to [0,255]
    real_img_6 = real_img_6.astype(np.int16) # set data type
    real_img_6 = np.transpose(real_img_6, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '6.png'), real_img_6)
    real_img_7 = img_real[6,:,:,:]*255 # restore to [0,255]
    real_img_7 = real_img_7.astype(np.int16) # set data type
    real_img_7 = np.transpose(real_img_7, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '7.png'), real_img_7)
    real_img_8 = img_real[7,:,:,:]*255 # restore to [0,255]
    real_img_8 = real_img_8.astype(np.int16) # set data type
    real_img_8 = np.transpose(real_img_8, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '8.png'), real_img_8)
    real_img_9 = img_real[8,:,:,:]*255 # restore to [0,255]
    real_img_9 = real_img_9.astype(np.int16) # set data type
    real_img_9 = np.transpose(real_img_9, (1, 2, 0))
    cv2.imwrite(os.path.join(path, '9.png'), real_img_9)
    # save fake images
    img_fake = img_fake/2 + 0.5
    img_fake = img_fake.cpu()
    img_fake = img_fake.data.squeeze().numpy()
    adv_img_0 = img_fake[9,:,:,:]*255 # restore to [0,255]
    adv_img_0 = adv_img_0.astype(np.int16) # set data type
    adv_img_0 = np.transpose(adv_img_0, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'0_adv.png'), adv_img_0)
    adv_img_1 = img_fake[0,:,:,:]*255 # restore to [0,255]
    adv_img_1 = adv_img_1.astype(np.uint8) # set data type
    adv_img_1 = np.transpose(adv_img_1, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'1_adv.png'), adv_img_1)
    adv_img_2 = img_fake[1,:,:,:]*255 # restore to [0,255]
    adv_img_2 = adv_img_2.astype(np.int16) # set data type
    adv_img_2 = np.transpose(adv_img_2, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'2_adv.png'), adv_img_2)   
    adv_img_3 = img_fake[2,:,:,:]*255 # restore to [0,255]
    adv_img_3 = adv_img_3.astype(np.int16) # set data type
    adv_img_3 = np.transpose(adv_img_3, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'3_adv.png'), adv_img_3)
    adv_img_4 = img_fake[3,:,:,:]*255 # restore to [0,255]
    adv_img_4 = adv_img_4.astype(np.int16) # set data type
    adv_img_4 = np.transpose(adv_img_4, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'4_adv.png'), adv_img_4)
    adv_img_5 = img_fake[4,:,:,:]*255 # restore to [0,255]
    adv_img_5 = adv_img_5.astype(np.int16) # set data type
    adv_img_5 = np.transpose(adv_img_5, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'5_adv.png'), adv_img_5)   
    adv_img_6 = img_fake[5,:,:,:]*255 # restore to [0,255]
    adv_img_6 = adv_img_6.astype(np.int16) # set data type
    adv_img_6 = np.transpose(adv_img_6, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'6_adv.png'), adv_img_6)   
    adv_img_7 = img_fake[6,:,:,:]*255 # restore to [0,255]
    adv_img_7 = adv_img_7.astype(np.int16) # set data type
    adv_img_7 = np.transpose(adv_img_7, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'7_adv.png'), adv_img_7)
    adv_img_8 = img_fake[7,:,:,:]*255 # restore to [0,255]
    adv_img_8 = adv_img_8.astype(np.int16) # set data type
    adv_img_8 = np.transpose(adv_img_8, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'8_adv.png'), adv_img_8) 
    adv_img_9 = img_fake[8,:,:,:]*255 # restore to [0,255]
    adv_img_9 = adv_img_9.astype(np.int16) # set data type
    adv_img_9 = np.transpose(adv_img_9, (1, 2, 0))
    cv2.imwrite(os.path.join(path,'9_adv.png'), adv_img_9) 
    
    print(n)
    return acc/n
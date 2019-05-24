import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_ssim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise2
from utils import perlin, colorize, toZeroThreshold
from math import log10
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
#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    return acc/n, ssim/n # returns attach success rate

def test_semitargeted(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
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

        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img.size(0)
#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    return acc/n, ssim/n # returns attach success rate

def test_semitargeted_targeted(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
    n = 0
    acc = 0
    ssim = 0
    target_pair = {0:5,1:8,2:8,3:5,4:9,5:3,6:2,7:9,8:5,9:4} # use class-wise L2 dict to evaluate
    
    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)
         # determine the corresponding target label
        y_target=[]
        #print(label)
        for x in label.numpy():
            #target_pair.get(x)
             y_target.append(target_pair.get(x))
        y_target = torch.LongTensor(y_target).to(device)
        acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img.size(0)
#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    return acc/n, ssim/n # returns attach success rate

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
    n = 0
    acc = 0
    ssim = 0
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
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img_real.size(0)
      
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

#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    return acc/n, ssim/n # returns attach success rate



def test_baseline_atnet(G, f, A, test_loader, epoch, epochs, device, verbose=True):
    n = 0
    acc = 0
    ssim = 0
    class_acc = np.zeros((1,10)) # count the number of success for each class
    

    G.eval()
    A.eval()
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
    plt.title('Fake image: digit %d'%(label[1]))
    plt.show()   
    
    
#    plt.figure(figsize=(1.5,1.5))
#    plt.imshow(img_real[3,:,:], cmap = 'gray')
#    plt.title('Real image: digit %d'%(label[1]))
#    plt.show()    
#    
#    plt.figure(figsize=(1.5,1.5))
#    plt.imshow(adversarial_img[3,:,:], cmap = 'gray')
#    plt.title('Real image: digit %d'%(label[1]))
#    plt.show()    

#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
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
    acc_class = np.divide(class_acc,class_num)
    n_pixel = img_real.size(2) * img_real.size(2)
    return acc/n, ssim/n, psnr/n, acc_class, distort_success/(n_success*n_pixel), distort_all/(n*n_pixel) # returns attach success rateclass_accclass_acc
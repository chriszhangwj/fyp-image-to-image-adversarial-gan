import torch
from torch.autograd import Variable
import pytorch_ssim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from noise import pnoise2
from utils import perlin, colorize

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
        pert = img_fake - img_real

        y_pred = f(img_fake)

        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img_real.size(0)
      
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
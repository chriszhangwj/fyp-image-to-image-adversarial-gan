import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import models
from prepare_dataset import load_dataset
from resnet import ResNet18

device = 'cuda'

# load target model
net = ResNet18()
checkpoint_path = os.path.join('saved', 'cifar10', 'target_models', 'best_cifar10.pth.tar')
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['state_dict'])
net.eval()
net.cuda()

train_data, test_data, in_channels, num_classes = load_dataset('cifar10')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

def test_cw(model, test_loader, device):
    n = 0
    acc = 0
    path = 'images/cifar10/cw'
    for i, (img, label) in enumerate(test_loader):
#         if i==100:
#             break
#         else:
#             img = img/2 + 0.5 # [0,1]
#             print('attacking image ', i)
#             img_fake = cw_l2(model, img, label, kappa=5, c=5, max_iter=500)
#             img_fake = img_fake*2 - 1 # [-1,1]
#             y_true = Variable(label.to(device))
#             y_pred = net(img_fake)
#             acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
#             n += img.size(0)
            
        if (i >= 9984) and (i<9994):
             print('attacking image ', i) #img is [-1,1]
             img = img/2 + 0.5 # [0,1]
             img_fake = cw_l2(model, img, label, kappa=5, c=5, max_iter=500) # [0,1]
             img_fake_temp = img_fake*2 - 1 # [-1,1]
             y_true = Variable(label.to(device))
             y_pred = net(img_fake_temp)
             acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
             n += img.size(0)
             img_real = img.cpu() # [0,1]
             img_real = img_real.data.squeeze().numpy()
             img_real = np.transpose(img_real,(1,2,0))
             real_img = img_real*255 # restore to [0,255]
             real_img = real_img.astype(np.float64) # set data type
             if i == 9993:
                cv2.imwrite(os.path.join(path, '0.png'), real_img)
             else:
                cv2.imwrite(os.path.join(path, '%d.png'%(i-9983)), real_img)

             #img_fake = img_fake/2 + 0.5 # [0,1]
             img_fake = img_fake.cpu()
             img_fake = img_fake.data.squeeze().numpy()
             fake_img = img_fake*255 # restore to [0,255]
             fake_img = fake_img.astype(np.float64) # set data type
             fake_img = np.transpose(fake_img,(1,2,0))
             
             
             
             if i == 9993:
                cv2.imwrite(os.path.join(path,'0_adv.png'), fake_img)
             else:
                cv2.imwrite(os.path.join(path,'%d_adv.png'%(i-9983)), fake_img)
    
    print('accuracy', acc/n)
 
    return acc/n 

# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
def cw_l2(model, images, labels, c=1e-6, kappa=0, max_iter=400, learning_rate=0.01) :
    images = images.to(device)     
    labels = labels.to(device)
    # Define f-function
    def f(x) :
        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        
        # If untargeted, optimize for making the other class most likely 
        return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10
    
    for step in range(max_iter) :
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
    
    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images

test_cw(net, test_loader, device)
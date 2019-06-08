import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import target_models
import resnet
import torch.optim as optim
from generators import Generator_CIFAR10 as Generator
from discriminators import Discriminator_CIFAR10 as Discriminator
from prepare_dataset import load_dataset
from train_function import train, train_plot, train_plot_cifar10
from test_function import test_cifar10
from resnet import ResNet18

import cv2
import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt

def CWLoss(logits, target, is_targeted, num_classes=10, kappa=0):
    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target_one_hot = torch.eye(num_classes).type(logits.type())[target.long()] # one hot vector for the target label
    #print(logits.size())
    # workaround here.
    # subtract large value from target class to find other max value
    # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    real = torch.sum(target_one_hot*logits, 1) # element-wise multiplication by *
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if is_targeted:
        return torch.sum(torch.max(other-real, kappa))
    return torch.sum(torch.max(real-other, kappa))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train AdvGAN')
    parser.add_argument('--epochs', type=int, default=15, required=False, help='no. of epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4, required=False, help='no. of workers (default: 4)')
    parser.add_argument('--target', type=int, required=False, help='Target label')
    parser.add_argument('--thres', type=float, required=False, default=0.3, help='Perturbation bound')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU?')

    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    target = args.target
    thres = args.thres # thres is hard-coded below, change it
    gpu = args.gpu

    dataset_name = 'cifar10'

    # alternatively set parameters here
    target = -1
    lr = 0.001 # original 0.001
    epochs = 20
    
    is_targeted = False
    if target in range(0, 10):
        is_targeted = True # bool variable to indicate targeted or untargeted attack

    print('Training AdvGAN ', '(Target %d)'%(target) if is_targeted else '(Untargeted)')
    
    train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    D = Discriminator()
    G = Generator()
    net = ResNet18()

    checkpoint_path = os.path.join('saved', 'cifar10', 'target_models', 'best_cifar10.pth.tar')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    if gpu:
        D.cuda()
        G.cuda()
        net.cuda()

    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    scheduler_G = StepLR(optimizer_G, step_size=5, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=5, gamma=0.5)

    criterion_adv =  CWLoss # loss for fooling target model
    criterion_gan = nn.MSELoss() # for gan loss
    alpha = 30 # gan loss multiplication factor
    beta = 15 # for hinge loss
    num_steps = 50 # number of generator updates for 1 discriminator update
    thres = c = 0.1 # perturbation bound, used in loss_hinge

    device = 'cuda' if gpu else 'cpu'
    loss_adv_epoch = np.array([]).reshape(0,1)
    loss_gan_epoch = np.array([]).reshape(0,1)
    loss_hinge_epoch = np.array([]).reshape(0,1)
    loss_g_epoch = np.array([]).reshape(0,1)
    loss_d_epoch = np.array([]).reshape(0,1)

    for epoch in range(epochs):
        acc_train, loss_adv_hist, loss_gan_hist, loss_hinge_hist, loss_g_hist, loss_d_hist = train_plot_cifar10(G, D, net, target, is_targeted, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps, verbose=True)
        acc_test, _ = test_cifar10(G, net, target, is_targeted, thres, test_loader, epoch, epochs, device, verbose=True)
        
        loss_adv_epoch=np.vstack([loss_adv_epoch, loss_adv_hist])
        loss_gan_epoch=np.vstack([loss_gan_epoch, loss_gan_hist])
        loss_hinge_epoch=np.vstack([loss_hinge_epoch, loss_hinge_hist])
        loss_g_epoch=np.vstack([loss_g_epoch, loss_g_hist])
        loss_d_epoch=np.vstack([loss_d_epoch, loss_d_hist])

        scheduler_G.step()
        scheduler_D.step()

        print("     "*20, end="\r")
        print('Epoch [%d/%d]\t\t\t'%(epoch+1, epochs))
        print('Train Acc: %.5f'%(acc_train))
        print('Test Acc: %.5f'%(acc_test))
        print('\n')

        torch.save({"epoch": epoch+1,
                    "epochs": epochs,
                    "is_targeted": is_targeted,
                    "target": target,
                    "thres": thres,
                    "state_dict": G.state_dict(),
                    "acc_test": acc_test,
                    "optimizer": optimizer_G.state_dict()
                    }, "saved/cifar10/advgan/advgan.pth.tar")
    
    # plot training curve
    fig, ax = plt.subplots()    
    ax.plot(loss_adv_epoch, label='loss_adv')
    ax.plot(loss_gan_epoch, label='loss_gan')
    ax.plot(loss_hinge_epoch, label='loss_pert')
    ax.set(xlabel='Steps (Number of batches)', ylabel='Magnitude',title='Loss evolution')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major',linestyle='-')
    ax.grid(which='minor',linestyle=':')
    plt.ylim((0,100))
    plt.legend(loc='upper right')
    plt.show()
    
    fig, ax = plt.subplots()    
    ax.plot(loss_gan_epoch, label='loss_gan')
    ax.plot(loss_d_epoch, label='loss_d')
    ax.set(xlabel='Steps (Number of batches)', ylabel='Magnitude',title='Loss evolution')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major',linestyle='-')
    ax.grid(which='minor',linestyle=':')
    plt.ylim((0,2))
    plt.legend(loc='upper right')
    plt.show()
    
    
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import target_models
from generators import Generator_CIFAR10 as Generator
from discriminators import Discriminator_ACGAN2_CIFAR10
from prepare_dataset import load_dataset
from train_function import train_baseline_CIFAR10
from test_function import test_baseline_CIFAR10
from utils import tile_evolution, plot_pert, tile_evolution_20, plot_pert_cifar10, tile_evolution_cifar10
from resnet import ResNet18

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
torch.manual_seed(0)

def CWLoss(logits, target, is_targeted, num_classes=10, kappa=0):
    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target_one_hot = torch.eye(num_classes).type(logits.type())[target.long()] # one hot vector for the target label
    real = torch.sum(target_one_hot*logits, 1) # element-wise multiplication by *
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)
    if is_targeted:
        return torch.sum(torch.max(other-real, kappa))
    return torch.sum(torch.max(real-other, kappa))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AdvGAN')
    parser.add_argument('--model', type=str, default="Model_C", required=False, choices=["Model_A", "Model_B", "Model_C"], help='model name (default: Model_C)')
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
    model_name = args.model
    target = args.target
    thres = args.thres # thres is hard-coded below, change itpi
    gpu = args.gpu

    dataset_name = 'cifar10'
    lr = 0.001 # original 0.001
    epochs = 15

    print('Training AdvGAN (Untargeted)')

    train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #D = Discriminator_ACGAN()
    D = Discriminator_ACGAN2_CIFAR10()
    G = Generator()
    net = ResNet18()
    net.cuda()

    checkpoint_path = os.path.join('saved', 'cifar10', 'target_models', 'best_cifar10.pth.tar')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    if gpu:
        D.cuda()
        G.cuda()

    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.5)

    criterion_adv =  CWLoss # loss for fooling target model
    criterion_gan = nn.MSELoss() # for gan loss
    criterion_aux = nn.CrossEntropyLoss() # for aux classifier; we do not use NLL loss
    alpha = 10 # gan loss multiplication factor
    beta = 0.7 # for adv loss
    gamma = 1 # for hinge loss
    lam = 0
    num_steps = 100 # number of generator updates for 1 discriminator update

    device = 'cuda' if gpu else 'cpu'
    loss_adv_epoch = np.array([]).reshape(0,1)
    loss_gan_epoch = np.array([]).reshape(0,1)
    loss_hinge_epoch = np.array([]).reshape(0,1)
    loss_g_epoch = np.array([]).reshape(0,1)
    loss_d_epoch = np.array([]).reshape(0,1)
    loss_real_epoch = np.array([]).reshape(0,1)
    loss_fake_epoch = np.array([]).reshape(0,1)
    loss_aux_epoch = np.array([]).reshape(0,1)
    

    for epoch in range(epochs):
        acc_train, loss_adv_hist, loss_gan_hist, loss_hinge_hist, loss_g_hist, loss_d_hist, loss_real_hist, loss_fake_hist, loss_aux_hist = train_baseline_CIFAR10(G, D, net, criterion_adv, criterion_gan, criterion_aux, alpha, beta, gamma, lam, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps, verbose=True)
        acc_test, _ = test_baseline_CIFAR10(G, net, test_loader, epoch, epochs, device, verbose=True)
        
        loss_adv_epoch=np.vstack([loss_adv_epoch, loss_adv_hist])
        loss_gan_epoch=np.vstack([loss_gan_epoch, loss_gan_hist])
        loss_hinge_epoch=np.vstack([loss_hinge_epoch, loss_hinge_hist])
        loss_g_epoch=np.vstack([loss_g_epoch, loss_g_hist])
        loss_d_epoch=np.vstack([loss_d_epoch, loss_d_hist])
        loss_real_epoch=np.vstack([loss_real_epoch, loss_real_hist])
        loss_fake_epoch=np.vstack([loss_fake_epoch, loss_fake_hist])
        loss_aux_epoch=np.vstack([loss_aux_epoch, loss_aux_hist])

        scheduler_G.step()
        scheduler_D.step()

        print("     "*20, end="\r")
        print('Epoch [%d/%d]\t\t\t'%(epoch+1, epochs))
        print('Train Acc: %.5f'%(acc_train))
        print('Test Acc: %.5f'%(acc_test))
        print('\n')

        torch.save({"epoch": epoch+1,
                    "epochs": epochs,
                    "thres": thres,
                    "state_dict": G.state_dict(),
                    "acc_test": acc_test,
                    "optimizer": optimizer_G.state_dict()
                    }, "saved/cifar10/ours/ours.pth.tar")
    
    # plot training curve
    fig, ax = plt.subplots()    
    ax.plot(loss_adv_epoch, label='loss_adv')
    ax.plot(loss_gan_epoch, label='loss_g')
    ax.plot(loss_hinge_epoch, label='loss_pert')
    ax.set(xlabel='Steps (Number of batches)', ylabel='Magnitude',title='Loss evolution')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major',linestyle='-')
    ax.grid(which='minor',linestyle=':')
    plt.legend(loc='upper right')
    #plt.ylim((0,100))
    plt.show()
    
    fig, ax = plt.subplots()    
    ax.plot(loss_gan_epoch, label='loss_g')
    ax.plot(loss_d_epoch, label='loss_d')
    ax.set(xlabel='Steps (Number of batches)', ylabel='Magnitude',title='Loss evolution')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major',linestyle='-')
    ax.grid(which='minor',linestyle=':')
    #plt.ylim((0,1.5))
    plt.legend(loc='upper right')
    plt.show()
    
    fig, ax = plt.subplots()    
    ax.plot(loss_real_epoch, label='loss_real')
    ax.plot(loss_fake_epoch, label='loss_fake')
    ax.set(xlabel='Steps (Number of batches)', ylabel='Magnitude',title='Loss evolution')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major',linestyle='-')
    ax.grid(which='minor',linestyle=':')
    plt.ylim((0,1))
    plt.legend(loc='upper right')
    plt.show()
    
    plot_pert_cifar10()
    tile_evolution_cifar10()
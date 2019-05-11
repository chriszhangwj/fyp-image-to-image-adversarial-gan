# Note that for semi-targeted attack, there is no need to re-train the target model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import target_models
from generators import Generator_MNIST as Generator
from discriminators import Discriminator_MNIST as Discriminator
from prepare_dataset import load_dataset
from train_function import train_semitargeted
from test_function import test_semitargeted

import cv2
import numpy as np
import os
import argparse

def CWLoss(logits, target, num_classes=10, kappa=0):
    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target_one_hot = []
    #print(target.size()) # torch.Size([128])
    #print(logits.size()) # torch.Size([128,10])
    #print(target.type()) # torch.cuda.LongTensor
    #print(logits.type()) # torch.cuda.FloatTensor
    for x in target.cpu().numpy():
        temp = torch.eye(num_classes).type(logits.type())[x]
        #print(np.shape(temp)) # torch.Size([10])
        #print(temp.type()) # torch.cuda.FloatTensor
        #print(temp.type()) # torch.cuda.FloatTensor
        temp = temp.cpu()
        #print(temp.type()) # torch.FloatTensor
        temp = temp.numpy()
        #print(type(temp)) # <class 'numpy.ndarray'>
        #print(np.shape(temp)) # (10,)
        target_one_hot.append(temp)        
    #print(target_one_hot)   
    
    # convert list to tensor
#    target_one_hot = torch.FloatTensor(target_one_hot)
#    print(target_one_hot.type())# torch.FloatTensor
#    print(target_one_hot.size()) # torch.Size([128, 10])
    
    target_one_hot = torch.FloatTensor(target_one_hot).to(device)
    #print(target_one_hot.type())# torch.cuda.FloatTensor
    #print(target_one_hot.size()) # torch.Size([128, 10])
    #print(logits.type()) # torch.cuda.FloatTensor
    #print(logits.size()) # torch.Size([128, 10])
    
    #print(len(target_one_hot)) # 128
    real = torch.sum(target_one_hot*logits, 1)
    #temp2 = target_one_hot*logits
    #print(temp2.type()) # torch.cuda.FloatTensor
    #print(temp2.size()) # torch.Size([128, 10])
    #print(real.size()) # torch.Size([128])
    #temp3 = 1-target_one_hot
    #print(temp3.size()) # torch.Size([128, 10])
    
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    #print(other.size()) # torch.Size([128])
    kappa = torch.zeros_like(other).fill_(kappa)

    return torch.sum(torch.max(other-real, kappa))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train AdvGAN')
    parser.add_argument('--model', type=str, default="Model_C", required=False, choices=["Model_A", "Model_B", "Model_C"], help='model name (default: Model_C)')
    parser.add_argument('--epochs', type=int, default=15, required=False, help='no. of epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=128, required=False, help='batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4, required=False, help='no. of workers (default: 4)')
    parser.add_argument('--thres', type=float, required=False, default=0.3, help='Perturbation bound')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU?')

    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    num_workers = args.num_workers
    epochs = args.epochs
    model_name = args.model
#    thres = args.thres # thres is hard-coded below, change it
    gpu = args.gpu
    dataset_name = 'mnist'

    # alternatively set parameters here
    model = 'Model_C'
    is_targeted = True # semi targeted
  
    print('Training AdvGAN (Semitargeted)')
    
    train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    D = Discriminator()
    G = Generator()
    f = getattr(target_models, model_name)(in_channels, num_classes)

    checkpoint_path = os.path.join('saved', 'target_models', 'semitargeted', 'best_%s_mnist.pth.tar'%(model_name))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    f.load_state_dict(checkpoint["state_dict"])
    f.eval()

    if gpu:
        D.cuda()
        G.cuda()
        f.cuda()

    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    scheduler_G = StepLR(optimizer_G, step_size=5, gamma=0.1)
    scheduler_D = StepLR(optimizer_D, step_size=5, gamma=0.1)

    criterion_adv =  CWLoss # loss for fooling target model
    criterion_gan = nn.MSELoss() # for gan loss
    alpha = 1 # gan loss multiplication factor
    beta = 1 # for hinge loss
    num_steps = 3 # number of generator updates for 1 discriminator update
    thres = c = 0.3 # perturbation bound, used in loss_hinge

    device = 'cuda' if gpu else 'cpu'

    for epoch in range(epochs):
        acc_train = train_semitargeted(G, D, f, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps, verbose=True)
        acc_test, _ = test_semitargeted(G, f, thres, test_loader, epoch, epochs, device, verbose=True)

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
                    }, "saved/target_models/semitargeted/generators/bound_%.1f/%s_%s.pth.tar"%(thres, model_name, 'semitargeted'))

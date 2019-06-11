import torch
from generators import Generator_CIFAR10 as Generator
from prepare_dataset import load_dataset
from test_function import eval_advgan_batch_cifar10, test_cifar10
from resnet import ResNet18

import cv2
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdvGAN for MNIST')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU?')

    args = parser.parse_args()
    gpu = args.gpu   
    
    thres = 0.09
    device = 'cuda'
    # load target model
    net = ResNet18()
    checkpoint_path = os.path.join('saved', 'cifar10', 'target_models', 'best_cifar10.pth.tar')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    net.cuda()
    
    # load generator
    G = Generator()
    checkpoint_name_G = 'advgan.pth.tar'
    checkpoint_path_G = os.path.join('saved', 'cifar10','advgan', checkpoint_name_G)
    checkpoint_G = torch.load(checkpoint_path_G)
    G.load_state_dict(checkpoint_G['state_dict'])
    G.eval()
    G.cuda() 
        
    train_data, test_data, in_channels, num_classes = load_dataset('cifar10')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
    
    # ------------------------------------------evaluate advgan---------------------------------------
    acc_test  = eval_advgan_batch_cifar10(G, net, thres, test_loader, device, verbose=True)
    #acc_test, _ = test_cifar10(G, net, -1, False, thres, test_loader, 1, 1, device, verbose=True)
    
    print('Test Acc: %.5f'%(acc_test))
    #print('SSIM: %.5f'%(loss_ssim))
    #print('PSNR %.5f'%(loss_psnr))
    #print('Distortion for successful samples %.5f'%(distort_success))
    #print('Distortion for all samples %.5f'%(distort_all))


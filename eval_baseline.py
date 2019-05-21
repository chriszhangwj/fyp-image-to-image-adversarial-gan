import torch
import torch.nn.functional as F
import target_models
from generators import Generator_MNIST as Generator
from prepare_dataset import load_dataset
from train_function import train
from test_function import eval_baseline

import cv2
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdvGAN for MNIST')
    parser.add_argument('--model', type=str, default="Model_C", required=False, choices=["Model_A", "Model_B", "Model_C"], help='model name (default: Model_C)')
    parser.add_argument('--target', type=int, required=False, help='Target label')
    parser.add_argument('--bound', type=float, default=0.3, choices=[0.2, 0.3], required=False, help='Perturbation bound (0.2 or 0.3)')
    parser.add_argument('--img', type=str, default='images/0.jpg', required=False, help='Image to perturb')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU?')

    args = parser.parse_args()
    model_name = args.model
    target = args.target
    thres = args.bound
    img_path = args.img
    gpu = args.gpu   
    
    # alternatively set parameters here
    thres = 0.3
    model_name = 'Model_C'
    digit = 2
    target = -1

    is_targeted = False
    if target in range(0, 10):
        is_targeted = True

    # load target model
    f = getattr(target_models, model_name)(1, 10)
    checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name))
    checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
    f.load_state_dict(checkpoint_f["state_dict"])
    f.eval()

    # load corresponding generator
    G = Generator()
    checkpoint_name_G = '%s_untargeted.pth.tar'%(model_name)
    checkpoint_path_G = os.path.join('saved', 'baseline', checkpoint_name_G)
    #checkpoint_path_G = os.path.join('saved','Model_C_untargeted.pth.tar')
    checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
    G.load_state_dict(checkpoint_G['state_dict'])
    G.eval()
    
    # compute test attack success rate
    M=0
    device = 'cuda' if gpu else 'cpu'
    if gpu:
        G.cuda()
        f.cuda()
    batch_size=1
    train_data, test_data, in_channels, num_classes = load_dataset('mnist')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    acc_test, loss_ssim, acc_class = eval_baseline(G, f, M, test_loader, 1, 1, device, verbose=True)
    print('Test Acc: %.5f'%(acc_test))
    print('SSIM: %.5f'%(loss_ssim))


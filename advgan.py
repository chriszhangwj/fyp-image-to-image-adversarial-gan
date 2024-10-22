import torch
import torch.nn.functional as F
import target_models
from generators import Generator_MNIST as Generator
from prepare_dataset import load_dataset
from train_function import train
from test_function import test

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
    img_path = 'images/%d.jpg'%(digit)

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
    checkpoint_name_G = '%s_target_%d.pth.tar'%(model_name, target) if is_targeted else '%s_untargeted.pth.tar'%(model_name)
    checkpoint_path_G = os.path.join('saved', 'generators', 'bound_%.1f'%(thres), checkpoint_name_G)
    #checkpoint_path_G = os.path.join('saved','Model_C_untargeted.pth.tar')
    checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
    G.load_state_dict(checkpoint_G['state_dict'])
    G.eval()

    # load img and preprocess as required by f and G
    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # img_path is the path to the image to be perturbed; images/0.jpg by default
    img = orig.copy().astype(np.float32)
    img = img[None, None, :, :]/255.0 # normalise the image to [0,1]

    x = torch.from_numpy(img) # convert numpy array to a tensor
    pert = G(x).data.clamp(min=-thres, max=thres) # use the pre-trained G to produce perturbation
    x_adv = x + pert 
    x_adv = x_adv.clamp(min=0, max=1)

    adversarial_img = x_adv.data.squeeze().numpy() # convert tensor to numpy
    perturbation = pert.data.squeeze().numpy() # convert tensor to numpy

    # prediction before and after attack
    #prob_before, y_before = torch.max(F.softmax(f(x), 1), 1) # original implementation when softmax is not included in the model
    #prob_after, y_after = torch.max(F.softmax(f(x_adv), 1), 1) 
    prob_before, y_before = torch.max(f(x), 1)
    prob_after, y_after = torch.max(f(x_adv), 1)

    print('Prediction before attack: %d [Prob: %0.4f]'%(y_before.item(), prob_before.item()))
    print('After attack: %d [Prob: %0.4f]'%(y_after.item(), prob_after.item()))
    
    # compute test attack success rate
    device = 'cuda' if gpu else 'cpu'
    if gpu:
        G.cuda()
        f.cuda()
    batch_size=1
    train_data, test_data, in_channels, num_classes = load_dataset('mnist')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    acc_test, loss_ssim = test(G, f, target, is_targeted, thres, test_loader, 1, 1, device, verbose=True)
    print('Test Acc: %.5f'%(acc_test))
    print('SIMM: %.5f'%(loss_ssim))

#    while True:
#        cv2.imshow('Adversarial Image', adversarial_img)
##        cv2.imshow('Perturbation', perturbation)
##        cv2.imshow('Image', orig)
#
#        key = cv2.waitKey(10) & 0xFF
#        if key == 27: # if ESC is pressed
#            break
#        if key == ord('s'):
#            d = 0
#            adversarial_img = adversarial_img*255 # restore to [0,255]
#            adversarial_img = adversarial_img.astype(np.uint8) # set data type
#            if is_targeted == True:
#                cv2.imwrite('targeted_%d_%d_%d.png'%(digit, target, y_after.item()), adversarial_img)
#            if is_targeted == False:
#                cv2.imwrite('untargeted_%d_%d.png'%(digit, y_after.item()), adversarial_img)
#            break
#        
#cv2.destroyAllWindows()

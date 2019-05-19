import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import numpy as np
import cv2
import argparse
import os
import target_models
from target_models import Model_C
from prepare_dataset import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='images/img_3.jpg', help='path to image')
parser.add_argument('--y', type=int, required=False, help='Label')
parser.add_argument('--gpu', action="store_true", default=True)

args = parser.parse_args()
image_path = args.img
y_true = args.y
gpu = args.gpu
IMG_SIZE = 28
digit = 3
image_path = 'images/digits/%d.jpg'%(digit)

def nothing(x):
    pass

def test_ce(model, eps, criterion, test_loader, device):
    n = 0
    acc = 0
    for i, (img, label) in enumerate(test_loader):
        inp = Variable(img.to(device).float(), requires_grad=True)
        out = model(inp)
        loss = criterion(out, Variable(torch.Tensor([float(label)]).to(device).long()))
        loss.backward()
        inp.data = inp.data + (eps * torch.sign(inp.grad.data))
        inp.data = inp.data.clamp(min=-1, max=1)
        inp.grad.data.zero_() # unnecessary
        y_pred = model(inp)
        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() 
        n += img.size(0)
    return acc/n 

def test_original(model, eps, criterion, test_loader, device):
    n = 0
    acc = 0
    for i, (img, label) in enumerate(test_loader):
        inp = Variable(img.to(device).float(), requires_grad=True)
        y_pred = model(inp)
        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() 
        n += img.size(0)
    return acc/n 

# load target model
model_name = "Model_C"
f = getattr(target_models, model_name)(1, 10)
saved = torch.load('best_Model_C_mnist.pth.tar', map_location='cpu')
#saved = torch.load('9920.pth.tar', map_location='cpu')
f.load_state_dict(saved['state_dict'])
f.eval()
f.cuda()

device = 'cuda' if gpu else 'cpu'
criterion = nn.CrossEntropyLoss()
eps=0.7
train_data, test_data, in_channels, num_classes = load_dataset('mnist')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
acc_test = test_ce(f, eps, criterion, test_loader, device)
print('Test Acc: %.5f'%(acc_test))

orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
img = orig.copy().astype(np.float32)
perturbation = np.empty_like(orig)

mean = [0.5]
std = [0.5]
img /= 255.0
img = (img - mean)/std
# prediction before attack
inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
out = f(inp)
pred = np.argmax(out.data.cpu().numpy())
print('Prediction before attack: %s' %(pred))
loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))
loss.backward()
inp.data = inp.data + eps * torch.sign(inp.grad.data)
inp.data = inp.data.clamp(min=-1, max=1)
 # predict on the adversarial image
pred_adv = np.argmax(f(inp).data.cpu().numpy())
print("After attack: eps [%f] \t%s"
%(eps, pred_adv), end="\r")#, end='\r')#'eps:', eps, end='\r')
# deprocess image
adv = inp.data.cpu().numpy()[0][0]
perturbation = adv-img
adv = (adv * std) + mean
adv = adv * 255.0
adv = np.clip(adv, 0, 255).astype(np.uint8)
perturbation = perturbation*255
perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)


# Interacive 
#window_adv = 'adversarial image'
#cv2.namedWindow(window_adv)
#cv2.createTrackbar('eps', window_adv, 1, 255, nothing)
#while True:
#    # get trackbar position
#    eps = cv2.getTrackbarPos('eps', window_adv)
#
#    inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
#
#    out = f(inp)
#    loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))
#
#    # compute gradients
#    loss.backward()
#
#    inp.data = inp.data + ((eps/255.0) * torch.sign(inp.grad.data))
#    inp.data = inp.data.clamp(min=-1, max=1)
#    inp.grad.data.zero_() # unnecessary
#
#    # predict on the adversarial image
#    pred_adv = np.argmax(f(inp).data.cpu().numpy())
#    print(" "*60, end='\r') # to clear previous line, not an elegant way
#    print("After attack: eps [%f] \t%s"
#            %(eps/255.0, pred_adv), end="\r")#, end='\r')#'eps:', eps, end='\r')
#
#    # deprocess image
#    adv = inp.data.cpu().numpy()[0][0]
#    perturbation = adv-img
#    adv = (adv * std) + mean
#    adv = adv * 255.0
#    adv = np.clip(adv, 0, 255).astype(np.uint8)
#    perturbation = perturbation*255
#    perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)
#
#    # display images
#    cv2.imshow(window_adv, perturbation)
#    cv2.imshow('perturbation', adv)
#    key = cv2.waitKey(500) & 0xFF
#    if key == 27:
#        break
#    elif key == ord('s'):
#        cv2.imwrite('img_adv.png', adv)
#        cv2.imwrite('perturbation.png', perturbation)
#print()
#cv2.destroyAllWindows()

while True:
    cv2.imshow('Adversarial Image', adv)
#   cv2.imshow('Perturbation', perturbation)
#   cv2.imshow('Image', orig)

    key = cv2.waitKey(10) & 0xFF
    if key == 27: # if ESC is pressed
        break
    if key == ord('s'):
        d = 0
        cv2.imwrite('fgsm_%d_%d_%0.2f.png'%(digit, pred_adv, eps), adv)
        break
cv2.destroyAllWindows()

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from generators import Generator_MNIST as Generator
import target_models

m = nn.Softmax()
model_name = 'Model_C'
f = getattr(target_models, model_name)(1, 10)
checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name)) # do not use the temp version
checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
f.load_state_dict(checkpoint_f["state_dict"])
f.eval()


fig = plt.figure(figsize=(50,50))
k=1;
digit=1
path = 'images/train_evolution/%d/'%(digit)

img_real = '%d_epoch_%d.png'%(digit,0)
img_path = os.path.join(path, img_real)
img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
fig.add_subplot(1,8,1)
plt.imshow(img_real)
plt.axis('off')
plt.title('Real', fontsize=30)

img_fake = '%d_epoch_%d.png'%(digit,15)
img_path = os.path.join(path, img_fake)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
fig.add_subplot(1,8,2)
plt.imshow(img)
plt.axis('off')

img_temp = img
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,2)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')

img_temp = img
img_temp[12][9] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,3)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')


img_temp[13][9] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,4)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')

img_temp[12][10] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,5)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')


img_temp[14][9] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,6)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')

img_temp[14][10] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,7)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')


img_temp[15][10] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,8)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')


fig = plt.figure(figsize=(50,50))
k=1;
digit=6
path = 'images/train_evolution/%d/'%(digit)

img_real = '%d_epoch_%d.png'%(digit,0)
img_path = os.path.join(path, img_real)
img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
fig.add_subplot(1,8,1)
plt.imshow(img_real)
plt.axis('off')
plt.title('Real', fontsize=30)

img_fake = '%d_epoch_%d.png'%(digit,15)
img_path = os.path.join(path, img_fake)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
fig.add_subplot(1,8,2)
plt.imshow(img)
plt.axis('off')

img_temp = img
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,2)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')

img_temp = img
img_temp[7][9] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,3)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')


img_temp[7][10] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,4)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')

img_temp[7][11] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,5)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')


img_temp[7][12] = 0
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,6)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')

img_temp[16][9] = 255
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,7)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')


img_temp[16][10] = 255
img_input = torch.from_numpy(img_temp).float()
img_input = torch.unsqueeze(img_input,0)/255.0
img_input = torch.unsqueeze(img_input,0)
y_pred = f(img_input) # obtain logits
y_prob = m(y_pred) # obtain softmax probabilities
y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
fig.add_subplot(1,8,8)
plt.imshow(img_temp)
plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30)
plt.axis('off')
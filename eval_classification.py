import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from generators import Generator_MNIST as Generator
from utils import cw_l2
import target_models

def tile_evolution_pred(digit=0):
    # load target model
    model_name = 'Model_C'
    f = getattr(target_models, model_name)(1, 10)
    checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name)) # do not use the temp version
    checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
    f.load_state_dict(checkpoint_f["state_dict"])
    f.eval()
    # plot evolution plot with predicted results
    fig = plt.figure(figsize=(50,50))
    fig.subplots_adjust(hspace=1.0,wspace=0)
    k=1; # image for subfigure position
    m = nn.Softmax()
    path = 'images/train_evolution/%d/'%(digit)
    images = os.listdir(path)
    images.sort()
            
    for j in range(21): # loop epochs
        # read images
        img_fake = '%d_epoch_%d.png'%(digit,j)
        img_path = os.path.join(path, img_fake)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # get prediction
        img_input = torch.from_numpy(img).float()
        img_input = torch.unsqueeze(img_input,0)/255.0
        img_input = torch.unsqueeze(img_input,0)
        img_input = img_input
        y_pred = f(img_input) # obtain logits
        y_prob = m(y_pred) # obtain softmax probabilities
        y_pred_label = torch.max(y_prob, 1)[1] # obtain predicted label
        y_pred_prob = y_prob[0][y_pred_label] # obtain confidence
        fig.add_subplot(1,21,k)
        k+=1
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        if j==0:
            plt.title('Real', fontsize=20)
        if j!=0:
            plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=20) 

#tile_evolution_pred(digit=0)
    
# generate adversaries for random images from test set
# load target model
#model_name = 'Model_C'
#f = getattr(target_models, model_name)(1, 10)
#checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name)) # do not use the temp version
#checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
#f.load_state_dict(checkpoint_f["state_dict"])
#f.eval()
##
## load generator
#G = Generator()
#checkpoint_name_G = '%s_untargeted.pth.tar'%(model_name)
#checkpoint_path_G = os.path.join('saved', 'baseline', checkpoint_name_G)
##checkpoint_path_G = os.path.join('saved','Model_C_untargeted.pth.tar')
#checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
#G.load_state_dict(checkpoint_G['state_dict'])
#G.eval()

# plot evolution plot with predicted results
#k=1; # image for subfigure position
#images = os.listdir('images/random_test/group3/')
#images.sort()
#fig = plt.figure(figsize=(50,50))
#m = nn.Softmax()
#
#for i, image in enumerate(images):
#    print(image)
#    img_path = os.path.join('images/random_test/group3/', image)
#    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#    img = torch.from_numpy(img).float()
#    img = torch.unsqueeze(img,0)/255.0
#    img_real = torch.unsqueeze(img,0)
#    img_fake = G(img_real).clamp(min=0, max=1) # use the pre-trained G to produce perturbation
#    y_pred = f(img_fake) # obtain logits
#    y_prob = m(y_pred)
#    y_pred_label = torch.max(y_pred, 1)[1] # obtain predicted label
#    y_pred_prob = y_prob[0][y_pred_label]
#    fig.add_subplot(1,11,i+1)
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = img_fake.detach().numpy()
#    plt.imshow(img_fake, cmap='gray')
#    plt.axis('off')
#    plt.tight_layout()
#    plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30) 
    

# generate adversaries for random images from test set - AdvGAN--------------------------------------------
# load target model
#model_name = 'Model_C'
#f = getattr(target_models, model_name)(1, 10)
#checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name)) # do not use the temp version
#checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
#f.load_state_dict(checkpoint_f["state_dict"])
#f.eval()
#G = Generator()
#checkpoint_name_G = 'advgan_%s_thres2.pth.tar'%(model_name)
#checkpoint_path_G = os.path.join('saved', 'advgan', checkpoint_name_G)
##checkpoint_path_G = os.path.join('saved', 'generators','bound_0.2', 'Model_C_untargeted.pth.tar')
#checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
#G.load_state_dict(checkpoint_G['state_dict'])
#G.eval()


# plot evolution plot with predicted results - AdvGAN
#k=1; # image for subfigure position
#images = os.listdir('images/random_test/group3/')
#images.sort()
#fig = plt.figure(figsize=(50,50))
#m = nn.Softmax()
#
#for i, image in enumerate(images):
#    print(image)
#    img_path = os.path.join('images/random_test/group3/', image) 
#    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#    img = orig.copy().astype(np.float32)
#    img = torch.from_numpy(img)
#    img = torch.unsqueeze(img,0)/255.0
#    img = torch.unsqueeze(img,0)
#    pert = G(img).data.clamp(min=-0.2, max=0.2)
#    img_fake = img+pert
#    y_pred = f(img_fake) # obtain logits
#    y_prob = m(y_pred)
#    y_pred_label = torch.max(y_pred, 1)[1] # obtain predicted label
#    y_pred_prob = y_prob[0][y_pred_label]
#    fig.add_subplot(1,11,i+1)
#    img_fake = img_fake.clamp(min=0, max=1) # use the pre-trained G to produce perturbation
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = img_fake.detach().numpy()
#    img_fake = img_fake * 255.0
#    img_fake = img_fake.astype(np.int16)
#    plt.imshow(img_fake, cmap='gray')
#    plt.axis('off')
#    plt.tight_layout()
#    plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30) 
#    
# generate adversaries for random images from test set - CW--------------------------------------------
# load target model
#model_name = 'Model_C'
#f = getattr(target_models, model_name)(1, 10)
#checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name)) # do not use the temp version
#checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
#f.load_state_dict(checkpoint_f["state_dict"])
#f.eval()
#f.cuda()
#images = os.listdir('images/random_test/group3/')
##label = torch.tensor([0,7,2,6,5,5,8,9,3,6,4]) # group 1
##label = torch.tensor([5,5,9,2,6,5,8,8,9,8,4]) # group 2
#label = torch.tensor([1,5,4,8,6,6,5,7,7,4,2]) # group 3
#images.sort()
#fig = plt.figure(figsize=(50,50))
#m = nn.Softmax()
#for i, image in enumerate(images):
#    print(image)
#    img_path = os.path.join('images/random_test/group3/', image) 
#    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#    img = orig.copy().astype(np.float32)
#    img = torch.from_numpy(img)
#    img = torch.unsqueeze(img,0)/255.0
#    img = torch.unsqueeze(img,0)
#    img_fake = cw_l2(f, img, label[i], kappa=0, c=4, max_iter=150)
#    y_pred = f(img_fake) # obtain logits
#    y_prob = m(y_pred)
#    y_pred_label = torch.max(y_pred, 1)[1] # obtain predicted label
#    y_pred_prob = y_prob[0][y_pred_label]
#    fig.add_subplot(1,11,i+1)
#    img_fake = img_fake.clamp(min=0, max=1) # use the pre-trained G to produce perturbation
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = img_fake.cpu().detach().numpy()
#    img_fake = img_fake * 255.0
#    img_fake = img_fake.astype(np.int16)
#    plt.imshow(img_fake, cmap='gray')
#    plt.axis('off')
#    plt.tight_layout()
#    plt.title('Pred: %d \n Conf: %.1f%%'%(y_pred_label, y_pred_prob*100), fontsize=30) 


    
## load target model
#model_name = 'Model_C'
#f = getattr(target_models, model_name)(1, 10)
#checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name)) # do not use the temp version
#checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
#f.load_state_dict(checkpoint_f["state_dict"])
#f.eval()
#
## load generator
#G = Generator()
#checkpoint_name_G = '%s_untargeted.pth.tar'%(model_name)
#checkpoint_path_G = os.path.join('saved', 'baseline', checkpoint_name_G)
##checkpoint_path_G = os.path.join('saved','Model_C_untargeted.pth.tar')
#checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
#G.load_state_dict(checkpoint_G['state_dict'])
#G.eval()
#
## plot evolution plot with predicted results
#k=1; # image for subfigure position
#images = os.listdir('images/mnist_test_png/')
#images.sort()
#for i, image in enumerate(images):
#    if i%1000==0:
#        print(i)
#    img_path = os.path.join('images/mnist_test_png/', image)
#    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#    img = torch.from_numpy(img).float()
#    img = torch.unsqueeze(img,0)/255.0
#    img_real = torch.unsqueeze(img,0)
#    img_fake = G(img_real).clamp(min=0, max=1) # use the pre-trained G to produce perturbation
#    y_pred = f(img_fake) # obtain logits
#    y_pred_label = torch.max(y_pred, 1)[1] # obtain predicted label
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = img_fake.detach().numpy()
#    cv2.imwrite('images/mnist_test_adv/%d.png'%(i) ,img_fake*255)
#    plt.axis('off')
#    plt.tight_layout()
#    plt.title('Pred: %d'%(y_pred_label), fontsize=20) 



# advgan generation
# load target model
#model_name='Model_C'
#f = getattr(target_models, model_name)(1, 10)
#checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name))
#checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
#f.load_state_dict(checkpoint_f["state_dict"])
#f.eval()
#
#G = Generator()
#checkpoint_name_G = 'advgan_%s_thres2.pth.tar'%(model_name)
#checkpoint_path_G = os.path.join('saved', 'advgan', checkpoint_name_G)
##checkpoint_path_G = os.path.join('saved', 'generators','bound_0.2', 'Model_C_untargeted.pth.tar')
#checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
#G.load_state_dict(checkpoint_G['state_dict'])
#G.eval()
#
#images = os.listdir('images/mnist_test_png/')
#images.sort()
#for i, image in enumerate(images):
#    if i%1000==0:
#        print(i)
#    img_path = os.path.join('images/mnist_test_png/', image)
#    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#    img = orig.copy().astype(np.float32)
#    img = torch.from_numpy(img)
#    img = torch.unsqueeze(img,0)/255.0
#    img = torch.unsqueeze(img,0)
#    pert = G(img).data.clamp(min=-0.2, max=0.2)
#    img_fake = img+pert
#    img_fake = img_fake.clamp(min=0, max=1) # use the pre-trained G to produce perturbation
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = torch.squeeze(img_fake,0)
#    img_fake = img_fake.detach().numpy()
#    img_fake = img_fake * 255.0
#    img_fake = img_fake.astype(np.int16)
#    img = img.detach().numpy()
#    img = img.squeeze()
#    img = img.squeeze()
#    cv2.imwrite('images/mnist_test_advgan_adv/%d.png'%(i) ,img_fake)
#   

# cw generation
# load target model
model_name='Model_C'
f = getattr(target_models, model_name)(1, 10)
checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist.pth.tar'%(model_name))
checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
f.load_state_dict(checkpoint_f["state_dict"])
f.eval()
f.cuda()

images = os.listdir('images/mnist_test_png/')
images.sort()
for i, image in enumerate(images):
    print(i)
    if i==1000:
        break
    else:
        img_path = os.path.join('images/mnist_test_png/', image)
        orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = orig.copy().astype(np.float32)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img,0)/255.0
        img = torch.unsqueeze(img,0)
        label= int((image.split('-')[1].split('.')[0])[3]) # extract label
        #print(label)
        #label = np.int8(label)
        label = torch.LongTensor(1, 1).fill_(label)
        #print(label)
        #print(type(label))
        img_fake = cw_l2(f, img, label, kappa=0, c=4, max_iter=200)
        #print(img_fake.size())
        img_fake = img_fake.cpu()
        img_fake = img_fake.data.detach().numpy()
        img_fake = img_fake*255
        img_fake = img_fake.squeeze()
        img_fake = img_fake.squeeze()
        img_fake = img_fake.astype(np.int16)
        #print(type(img_fake))
        #print(np.shape(img_fake))
        cv2.imwrite('images/mnist_test_cw_adv/%d.png'%(i) ,img_fake)


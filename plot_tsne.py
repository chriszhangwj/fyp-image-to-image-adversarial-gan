import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import target_models
import os
import torch
import torch.optim as optim
import torch.nn as nn
import time
import pytorch_ssim
from tensorflow.examples.tutorials.mnist import input_data
from torch.autograd import Variable
from prepare_dataset import load_dataset
from generators import Generator_MNIST as Generator
from utils import tsne_plot
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

def get_mnist(data_loader):
    image_vec = np.array([]).reshape(0,784)
    label_vec = np.array([]).reshape(0,1)
    for i, (img, label) in enumerate(data_loader):
        img = img.view(1,-1)
        label = label.view(1,-1)
        image_vec = np.vstack([image_vec,img])
        label_vec = np.vstack([label_vec,label])
        if i%100 == 0:
            print(i)
    return image_vec, label_vec


def get_adv(data_loader, attack, device):
    image_vec = np.array([]).reshape(0,784)
    label_vec = np.array([]).reshape(0,1)  
    if attack == 'advgan':
        G = Generator()
        checkpoint_name_G = 'advgan_Model_C_untargeted.pth.tar'
        checkpoint_path_G = os.path.join('saved', 'advgan', checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval()
        G.cuda()
        thres= 0.2
        for i, (img, label) in enumerate(data_loader):
            img_real = Variable(img.to(device))
            pert = torch.clamp(G(img_real), -thres, thres)
            img_fake = pert + img_real
            img_fake = img_fake.clamp(min=0, max=1)
            img_fake = img_fake.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy()
            label = label.view(1,-1)
            label = label.cpu().detach().numpy()
            image_vec = np.vstack([image_vec,img_fake])
            label_vec = np.vstack([label_vec,label])
            if i%100 == 0:
                print(i)
        
    if attack == 'fgsm':
        f = getattr(target_models, "Model_C")(1, 10)
        checkpoint_path_f = os.path.join('saved', 'target_models', 'best_Model_C_mnist_temp.pth.tar')
        checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
        f.load_state_dict(checkpoint_f["state_dict"])
        f.eval()
        f.cuda()
        criterion = nn.CrossEntropyLoss()
        eps=0.5
        for i, (img, label) in enumerate(data_loader):
            inp = Variable(img.to(device).float(), requires_grad=True)
            out = f(inp)
            loss = criterion(out, Variable(torch.Tensor([float(label)]).to(device).long()))
            loss.backward()
            img_fake = inp.data + (eps * torch.sign(inp.grad.data))
            img_fake = img_fake.clamp(min=-1, max=1)
            img_fake = img_fake.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy()
            label = label.view(1,-1)
            label = label.cpu().detach().numpy()
            inp.grad.data.zero_() # unnecessary
            image_vec = np.vstack([image_vec,img_fake])
            label_vec = np.vstack([label_vec,label])
            if i%100 == 0:
                print(i)   
                
    if attack == 'ours':
        G = Generator()
        checkpoint_name_G = 'Model_C_untargeted.pth.tar'
        checkpoint_path_G = os.path.join('saved', 'baseline', checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval()
        G.cuda()
        for i, (img, label) in enumerate(data_loader):
            img_real = Variable(img.to(device))
            img_fake = torch.clamp(G(img_real), 0, 1)
            img_fake = img_fake.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy()
            label = label.view(1,-1)
            label = label.cpu().detach().numpy()
            image_vec = np.vstack([image_vec,img_fake])
            label_vec = np.vstack([label_vec,label])
            if i%100 == 0:
                print(i)
    return image_vec, label_vec

def get_time(data_loader, attack, device):
    if attack == 'advgan':
        start = time.time()
        G = Generator()
        checkpoint_name_G = 'advgan_Model_C_untargeted.pth.tar'
        checkpoint_path_G = os.path.join('saved', 'advgan', checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval()
        G.cuda()
        thres= 0.2
        for i, (img, label) in enumerate(data_loader):
            img_real = Variable(img.to(device))
            pert = torch.clamp(G(img_real), -thres, thres)
            img_fake = pert + img_real
            img_fake = img_fake.clamp(min=0, max=1)
        end = time.time()
        print(end-start)
        
    if attack == 'fgsm':
        start = time.time()
        f = getattr(target_models, "Model_C")(1, 10)
        checkpoint_path_f = os.path.join('saved', 'target_models', 'best_Model_C_mnist_temp.pth.tar')
        checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
        f.load_state_dict(checkpoint_f["state_dict"])
        f.eval()
        f.cuda()
        criterion = nn.CrossEntropyLoss()
        eps=0.5
        for i, (img, label) in enumerate(data_loader):
            inp = Variable(img.to(device).float(), requires_grad=True)
            out = f(inp)
            loss = criterion(out, Variable(torch.Tensor([float(label)]).to(device).long()))
            loss.backward()
            img_fake = inp.data + (eps * torch.sign(inp.grad.data))
            img_fake = img_fake.clamp(min=-1, max=1)        
        end = time.time()
        print(end-start)

    if attack == 'ours':
        start = time.time()
        G = Generator()
        checkpoint_name_G = 'Model_C_untargeted.pth.tar'
        checkpoint_path_G = os.path.join('saved', 'baseline', checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval()
        G.cuda()
        for i, (img, label) in enumerate(data_loader):
            img_real = Variable(img.to(device))
            img_fake = torch.clamp(G(img_real), 0, 1)
        end = time.time()
        print(end-start)
        
        
        
def get_distort(data_loader, attack, device):
    if attack == 'advgan':
        G = Generator()
        checkpoint_name_G = 'advgan_Model_C_thres2_temp.pth.tar'
        checkpoint_path_G = os.path.join('saved', 'advgan', checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval()
        G.cuda()
        thres= 0.2
        dist_l0, dist_l1, dist_l2, dist_linf, ssim, psnr = 0, 0, 0, 0, 0, 0
        for i, (img, label) in enumerate(data_loader):
            img_real = Variable(img.to(device))
            pert = torch.clamp(G(img_real), -thres, thres)
            img_fake = pert + img_real
            img_fake = img_fake.clamp(min=0, max=1)
            img_fake = img_fake.view(1,-1)
            img_real = img_real.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy().squeeze()
            img_real = img_real.cpu().detach().numpy().squeeze()
            dist_l0 += np.linalg.norm(img_real-img_fake, 0)
            dist_l1 += np.linalg.norm(img_real-img_fake, 1)
            dist_l2 += np.linalg.norm(img_real-img_fake, 2)
            dist_linf += np.linalg.norm(img_real-img_fake, np.inf)
            ssim += compare_ssim(img_real, img_fake)
            psnr += compare_psnr(img_real,img_fake)
            if i%100 == 0:
                print(i)
        print('l0 distance: ', dist_l0/10000)
        print('l1 distance: ', dist_l1/10000)
        print('l2 distance: ', dist_l2/10000)
        print('linf distance: ', dist_linf/10000)
        print('ssim: ', ssim/10000)
        print('psnr: ', psnr/10000)
    if attack == 'fgsm':
        f = getattr(target_models, "Model_C")(1, 10)
        checkpoint_path_f = os.path.join('saved', 'target_models', 'best_Model_C_mnist_temp.pth.tar')
        checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
        f.load_state_dict(checkpoint_f["state_dict"])
        f.eval()
        f.cuda()
        criterion = nn.CrossEntropyLoss()
        eps=0.5
        dist_l0, dist_l1, dist_l2, dist_linf, ssim, psnr = 0, 0, 0, 0, 0, 0
        for i, (img, label) in enumerate(data_loader):
            inp = Variable(img.to(device).float(), requires_grad=True)
            out = f(inp)
            loss = criterion(out, Variable(torch.Tensor([float(label)]).to(device).long()))
            loss.backward()
            img_fake = inp.data + (eps * torch.sign(inp.grad.data))
            img_fake = img_fake.clamp(min=-1, max=1)
            img_fake = img_fake.view(1,-1)
            img_real = inp.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy().squeeze()
            img_real = img_real.cpu().detach().numpy().squeeze()
            dist_l0 += np.linalg.norm(img_real-img_fake, 0)
            dist_l1 += np.linalg.norm(img_real-img_fake, 1)
            dist_l2 += np.linalg.norm(img_real-img_fake, 2)
            dist_linf += np.linalg.norm(img_real-img_fake, np.inf)
            ssim += compare_ssim(img_real, img_fake)
            psnr += compare_psnr(img_real,img_fake)
            if i%100 == 0:
                print(i)
        print('l0 distance: ', dist_l0/10000)
        print('l1 distance: ', dist_l1/10000)
        print('l2 distance: ', dist_l2/10000)
        print('linf distance: ', dist_linf/10000)
        print('ssim: ', ssim/10000)
        print('psnr: ', psnr/10000)
                
    if attack == 'ours':
        G = Generator()
        checkpoint_name_G = 'Model_C_untargeted.pth.tar'
        checkpoint_path_G = os.path.join('saved', 'baseline', checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval()
        G.cuda()
        dist_l0, dist_l1, dist_l2, dist_linf, ssim, psnr = 0, 0, 0, 0, 0, 0
        for i, (img, label) in enumerate(data_loader):
            img_real = Variable(img.to(device))
            img_fake = torch.clamp(G(img_real), 0, 1)
            img_fake = img_fake.view(1,-1)
            img_real = img_real.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy().squeeze()
            img_real = img_real.cpu().detach().numpy().squeeze()
            dist_l0 += np.linalg.norm(img_real-img_fake, 0)
            dist_l1 += np.linalg.norm(img_real-img_fake, 1)
            dist_l2 += np.linalg.norm(img_real-img_fake, 2)
            dist_linf += np.linalg.norm(img_real-img_fake, np.inf)
            ssim += compare_ssim(img_real, img_fake)
            psnr += compare_psnr(img_real,img_fake)
            if i%100 == 0:
                print(i)     
        print('l0 distance: ', dist_l0/10000)
        print('l1 distance: ', dist_l1/10000)
        print('l2 distance: ', dist_l2/10000)
        print('linf distance: ', dist_linf/10000)
        print('ssim: ', ssim/10000)
        print('psnr: ', psnr/10000)
        
        
def get_indiv_class(data_loader, attack, device):
    if attack == 'advgan':
        G = Generator()
        checkpoint_name_G = 'advgan_Model_C_thres2_temp.pth.tar'
        checkpoint_path_G = os.path.join('saved', 'advgan', checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval()
        G.cuda()
        thres= 0.2
        class_l0_dist = np.zeros((1,10)) 
        class_l1_dist = np.zeros((1,10))
        class_l2_dist = np.zeros((1,10))
        class_linf_dist = np.zeros((1,10))
        class_ssim = np.zeros((1,10))
        class_psnr = np.zeros((1,10))
        class_num = np.zeros((1,10))
        for i, (img, label) in enumerate(data_loader):
            img_real = Variable(img.to(device))
            pert = torch.clamp(G(img_real), -thres, thres)
            img_fake = pert + img_real
            img_fake = img_fake.clamp(min=0, max=1)
            img_fake = img_fake.view(1,-1)
            img_real = img_real.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy().squeeze()
            img_real = img_real.cpu().detach().numpy().squeeze()
            y_true = Variable(label.to(device))
            class_num[0,y_true] = class_num[0,y_true]+1
            class_l0_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 0)  
            class_l1_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 1)
            class_l2_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 2)
            class_linf_dist[0,y_true] += np.linalg.norm(img_real-img_fake, np.inf)
            class_ssim[0,y_true] += compare_ssim(img_real, img_fake)
            class_psnr[0,y_true] += compare_psnr(img_real,img_fake)
            if i%100 == 0:
                print(i)
        class_l0_dist = np.divide(class_l0_dist,class_num)
        class_l1_dist = np.divide(class_l1_dist,class_num)
        class_l2_dist = np.divide(class_l2_dist,class_num)
        class_linf_dist = np.divide(class_linf_dist,class_num)
        class_ssim = np.divide(class_ssim,class_num)
        class_psnr = np.divide(class_psnr,class_num)
        
    if attack == 'fgsm':
        f = getattr(target_models, "Model_C")(1, 10)
        checkpoint_path_f = os.path.join('saved', 'target_models', 'best_Model_C_mnist_temp.pth.tar')
        checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
        f.load_state_dict(checkpoint_f["state_dict"])
        f.eval()
        f.cuda()
        criterion = nn.CrossEntropyLoss()
        eps=0.5
        class_l0_dist = np.zeros((1,10)) 
        class_l1_dist = np.zeros((1,10))
        class_l2_dist = np.zeros((1,10))
        class_linf_dist = np.zeros((1,10))
        class_ssim = np.zeros((1,10))
        class_psnr = np.zeros((1,10))
        class_num = np.zeros((1,10))        
        for i, (img, label) in enumerate(data_loader):
            inp = Variable(img.to(device).float(), requires_grad=True)
            out = f(inp)
            loss = criterion(out, Variable(torch.Tensor([float(label)]).to(device).long()))
            loss.backward()
            img_fake = inp.data + (eps * torch.sign(inp.grad.data))
            img_fake = img_fake.clamp(min=-1, max=1)
            img_fake = img_fake.view(1,-1)
            img_real = inp.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy().squeeze()
            img_real = img_real.cpu().detach().numpy().squeeze()
            y_true = Variable(label.to(device))
            class_num[0,y_true] = class_num[0,y_true]+1
            class_l0_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 0)  
            class_l1_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 1)
            class_l2_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 2)
            class_linf_dist[0,y_true] += np.linalg.norm(img_real-img_fake, np.inf)
            class_ssim[0,y_true] += compare_ssim(img_real, img_fake)
            class_psnr[0,y_true] += compare_psnr(img_real,img_fake)
            if i%100 == 0:
                print(i)
        class_l0_dist = np.divide(class_l0_dist,class_num)
        class_l1_dist = np.divide(class_l1_dist,class_num)
        class_l2_dist = np.divide(class_l2_dist,class_num)
        class_linf_dist = np.divide(class_linf_dist,class_num)
        class_ssim = np.divide(class_ssim,class_num)
        class_psnr = np.divide(class_psnr,class_num)
                
    if attack == 'ours':
        G = Generator()
        checkpoint_name_G = 'Model_C_untargeted.pth.tar'
        checkpoint_path_G = os.path.join('saved', 'baseline', checkpoint_name_G)
        checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
        G.load_state_dict(checkpoint_G['state_dict'])
        G.eval()
        G.cuda()
        class_l0_dist = np.zeros((1,10)) 
        class_l1_dist = np.zeros((1,10))
        class_l2_dist = np.zeros((1,10))
        class_linf_dist = np.zeros((1,10))
        class_ssim = np.zeros((1,10))
        class_psnr = np.zeros((1,10))
        class_num = np.zeros((1,10))        
        for i, (img, label) in enumerate(data_loader):
            img_real = Variable(img.to(device))
            img_fake = torch.clamp(G(img_real), 0, 1)
            img_fake = img_fake.view(1,-1)
            img_real = img_real.view(1,-1)
            img_fake = img_fake.cpu().detach().numpy().squeeze()
            img_real = img_real.cpu().detach().numpy().squeeze()
            y_true = Variable(label.to(device))
            class_num[0,y_true] = class_num[0,y_true]+1
            class_l0_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 0)  
            class_l1_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 1)
            class_l2_dist[0,y_true] += np.linalg.norm(img_real-img_fake, 2)
            class_linf_dist[0,y_true] += np.linalg.norm(img_real-img_fake, np.inf)
            class_ssim[0,y_true] += compare_ssim(img_real, img_fake)
            class_psnr[0,y_true] += compare_psnr(img_real,img_fake)
            if i%100 == 0:
                print(i)
        class_l0_dist = np.divide(class_l0_dist,class_num)
        class_l1_dist = np.divide(class_l1_dist,class_num)
        class_l2_dist = np.divide(class_l2_dist,class_num)
        class_linf_dist = np.divide(class_linf_dist,class_num)
        class_ssim = np.divide(class_ssim,class_num)
        class_psnr = np.divide(class_psnr,class_num)
    return class_l0_dist, class_l1_dist, class_l2_dist, class_linf_dist, class_ssim, class_psnr
            
train_data, test_data, in_channels, num_classes = load_dataset('mnist')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
#image_vec, label_vec = get_mnist(test_loader)
#np.savetxt("mnist_test_image_numpy.csv", image_vec, delimiter=",")
#np.savetxt("mnist_test_label_numpy.csv", label_vec, delimiter=",")

#image_vec = np.genfromtxt("mnist_test_image_numpy.csv", delimiter=',')
#label_vec = np.genfromtxt("mnist_test_label_numpy.csv", delimiter=',')

device = 'cuda'
#image_vec, label_vec = get_adv(test_loader,'ours', device)
#tsne_plot(image_vec, label_vec, 2)
#get_time(test_loader, 'ours', device)
#get_distort(test_loader, 'advgan', device)
l0_advgan, l1_advgan, l2_advgan, linf_advgan, ssim_advgan, psnr_advgan = get_indiv_class(test_loader, 'advgan', device)
l0_fgsm, l1_fgsm, l2_fgsm, linf_fgsm, ssim_fgsm, psnr_fgsm = get_indiv_class(test_loader, 'fgsm', device)
l0_ours, l1_ours, l2_ours, linf_ours, ssim_ours, psnr_ours = get_indiv_class(test_loader, 'ours', device)

l0_advgan = np.reshape(l0_advgan,(10,1))
l0_fgsm = np.reshape(l0_fgsm,(10,1))
l0_ours = np.reshape(l0_ours,(10,1))

l1_advgan = np.reshape(l1_advgan,(10,1))
l1_fgsm = np.reshape(l1_fgsm,(10,1))
#l1_ours = np.reshape(l1_ours,(10,1))

l2_advgan = np.reshape(l2_advgan,(10,1))
l2_fgsm = np.reshape(l2_fgsm,(10,1))
#l2_ours = np.reshape(l2_ours,(10,1))

linf_advgan = np.reshape(linf_advgan,(10,1))
linf_fgsm = np.reshape(linf_fgsm,(10,1))
#linf_ours = np.reshape(linf_ours,(10,1))

ssim_advgan = np.reshape(ssim_advgan,(10,1))
ssim_fgsm = np.reshape(ssim_fgsm,(10,1))
#ssim_ours = np.reshape(ssim_ours,(10,1))

psnr_advgan = np.reshape(psnr_advgan,(10,1))
psnr_fgsm = np.reshape(psnr_fgsm,(10,1))
#psnr_ours = np.reshape(psnr_ours,(10,1))

mnist_class = {'0','1','2','3','4','5','6','7','8','9'}
y_pos = np.arange(len(mnist_class))
bar_width = 0.35
opacity = 0.8

plt_fgsm = plt.bar(y_pos, l0_fgsm, bar_width, alpha=0.6, label='FGSM')
plt_advgan = plt.bar(y_pos + bar_width, l0_advgan, bar_width, alpha=0.6, label='AdvGAN')
plt_ours = plt.bar(y_pos + 2*bar_width, l0_ours, bar_width, alpha=0.6, label='Ours')

plt.xticks(y_pos + bar_width, ('0','1','2','3','4','5','6','7','8','9'))
plt.xlabel('Class')
plt.legend()
plt.tight_layout()
plt.show()


#plt_fgsm = plt.bar(y_pos, l1_fgsm, bar_width, align='center', alpha=0.6, label='FGSM')
#plt_advgan = plt.bar(y_pos + bar_width, l1_advgan, bar_width, align='center', alpha=0.6, label='AdvGAN')
#plt_ours = plt.bar(y_pos + 2*bar_width, l1_ours, bar_width, align='center', alpha=0.6, label='Ours')
#
#plt.xticks(y_pos + bar_width, ('0','1','2','3','4','5','6','7','8','9'))
#plt.xlabel('Class')
#plt.legend()
#plt.tight_layout()
#plt.show()


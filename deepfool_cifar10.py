import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from prepare_dataset import load_dataset
from resnet import ResNet18

SEED = 1

class Attacker:
    def __init__(self, clip_max=1, clip_min=0):
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        pass

class DeepFool(Attacker):
    def __init__(self, max_iter=50, clip_max=1, clip_min=0):
        super(DeepFool, self).__init__(clip_max, clip_min)
        self.max_iter = max_iter

    def generate(self, model, x, y, device):
        nx = Variable(x.to(device)) # image
        nx.requires_grad_()
        eta = torch.zeros(nx.shape) # perturbation
        eta = Variable(eta.to(device))
        
        temp = nx+eta # [0,1]
        temp = (temp-0.5) * 2
        out = model(temp)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()
        #print('py', py)
        #print('ny', ny)

        i_iter = 0
        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue
                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                fi = fi.cpu()
                wi = wi.cpu()
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi
                
            ri = Variable(ri.to(device))
            eta += ri.clone()
            nx.grad.data.zero_()
            temp = torch.clamp(nx+eta,0,1) # [0,1]
            temp = (temp-0.5)*2 # [-1,1]
            out = model(temp)
            py = out.max(1)[1].item()
            i_iter += 1
            if i_iter+1 == self.max_iter:
                print('failed to converge')
        
        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        return x_adv.detach()
    
# load target model
net = ResNet18()
checkpoint_path = os.path.join('saved', 'cifar10', 'target_models', 'best_cifar10.pth.tar')
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['state_dict'])
net.eval()
net.cuda()

device = 'cuda' 
train_data, test_data, in_channels, num_classes = load_dataset('cifar10')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    
attacker = DeepFool(max_iter=4, clip_max=1, clip_min=0)

n, acc = 0, 0
for i, (img, label) in enumerate(test_loader):
    img = img/2 + 0.5 # [0,1]
#    print('attacking', i)
#    if i == 1000:
#        break
#    else:
#        img_real = Variable(img.to(device))
#        #print(img_real)
#        y_true = Variable(label.to(device))
#        img_fake = attacker.generate(net, img_real, y_true, device) # [0,1]
#        img_fake = (img_fake-0.5)*2 # [-1,1]
#        #img_fake = img_fake+0.5
#        #print(img_fake)
#        y_pred = net(img_fake)
#        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
#        n += img.size(0)
        
    if (i >= 9984) and (i<9994):
        path = 'images/cifar10/deepfool'
        print('attacking image ', i)
        img_real = Variable(img.to(device))
        y_true = Variable(label.to(device))
        img_fake = attacker.generate(net, img_real, y_true, device)
        img_real = img_real.cpu()
        img_real = img_real.data.squeeze().numpy()
        img_real = np.transpose(img_real,(1,2,0))
        #img_real = img_real/2 + 0.5 # [0,1]
        real_img = img_real*255 # restore to [0,255]
        real_img = real_img.astype(np.float64) # set data type
        if i == 9993:
            cv2.imwrite(os.path.join(path, '0.png'), real_img)
        else:
            cv2.imwrite(os.path.join(path, '%d.png'%(i-9983)), real_img)
            
        img_fake = img_fake.cpu()
        img_fake = img_fake.data.squeeze().numpy()
        img_fake = np.transpose(img_fake,(1,2,0))
        fake_img = img_fake*255 # restore to [0,255]
        #fake_img = fake_img.astype(np.int16) # set data type
        if i == 9993:
            cv2.imwrite(os.path.join(path,'0_adv.png'), fake_img)
        else:
            cv2.imwrite(os.path.join(path,'%d_adv.png'%(i-9983)), fake_img)
        
#print(acc/n) 
       
#img_real = img_real.cpu()
#img_real = img_real.data.squeeze().numpy()
#plt.figure(figsize=(2,2))
#plt.imshow(img_real[:,:], cmap = 'gray')
#plt.title('Real image: digit %d'%(label))
#plt.show()   
#print('y_true: %d'%(y_true))
#
#img_fake = img_fake.cpu()
#adversarial_img = img_fake.data.squeeze().numpy()
#label = label.cpu()
#label = label.data.squeeze().numpy()
#plt.figure(figsize=(2,2))
#plt.imshow(adversarial_img[:,:],vmin=0, vmax=1, cmap = 'gray')
#plt.title('Real image: digit %d'%(label))
#plt.show()    
#print('y_pred: %d,'%(torch.max(y_pred, 1)[1]))
#path = 'images/cw'
#print('accuracy: ', acc/n)
    

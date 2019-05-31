import numpy as np
import torch
import torch.nn.functional as F
import os
import target_models
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from prepare_dataset import load_dataset

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
        
        out = model(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()
        print('py', py)
        print('ny', ny)

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
            temp = torch.clamp(nx+eta,0,1)
            out = model(temp)
            py = out.max(1)[1].item()
            i_iter += 1
            print('py', py)
            print('ny', ny)
        
        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        return x_adv.detach()
    
# load target model
model_name = 'Model_C'
f = getattr(target_models, model_name)(1, 10)
checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist_temp.pth.tar'%(model_name))
checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
f.load_state_dict(checkpoint_f["state_dict"])
f.eval()
f.cuda()

device = 'cuda' 
train_data, test_data, in_channels, num_classes = load_dataset('mnist')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    
attacker = DeepFool(max_iter=1000, clip_max=1, clip_min=0)

n, acc = 0, 0
for i, (img, label) in enumerate(test_loader):
    if i == 10:
        break
    else:
        img_real = Variable(img.to(device))
        #print(img_real)
        y_true = Variable(label.to(device))
        img_fake = attacker.generate(f, img_real, y_true, device)
        #img_fake = img_fake+0.5
        #print(img_fake)
        y_pred = f(img_fake)
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        n += img.size(0)
        
print(n)
print(acc) 
       
img_real = img_real.cpu()
img_real = img_real.data.squeeze().numpy()
plt.figure(figsize=(2,2))
plt.imshow(img_real[:,:], cmap = 'gray')
plt.title('Real image: digit %d'%(label))
plt.show()   
print('y_true: %d'%(y_true))

img_fake = img_fake.cpu()
adversarial_img = img_fake.data.squeeze().numpy()
label = label.cpu()
label = label.data.squeeze().numpy()
plt.figure(figsize=(2,2))
plt.imshow(adversarial_img[:,:],vmin=0, vmax=1, cmap = 'gray')
plt.title('Real image: digit %d'%(label))
plt.show()    
print('y_pred: %d,'%(torch.max(y_pred, 1)[1]))
path = 'images/cw'
print('accuracy: ', acc/n)
    

import numpy as np
import torch
import torch.nn.functional as F
import target_models
from torch.utils.data import DataLoader
from prepare_dataset import load_dataset

class BIM(Attacker):
    def __init__(self, eps=0.15, eps_iter=0.01, n_iter=50, clip_max=0.5, clip_min=-0.5):
        super(BIM, self).__init__(clip_max, clip_min)
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_iter = n_iter

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        ny = torch.unsqueeze(y, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        for i in range(self.n_iter):
            out = model(nx+eta)
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps_iter * torch.sign(nx.grad.data)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()

        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()
    
# load target model
model_name = "Model_C"
f = getattr(target_models, model_name)(1, 10)
saved = torch.load('best_Model_C_mnist.pth.tar', map_location='cpu')
#saved = torch.load('9920.pth.tar', map_location='cpu')
f.load_state_dict(saved['state_dict'])
f.eval()
f.cuda()

device = 'cuda' 
train_data, test_data, in_channels, num_classes = load_dataset('mnist')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
    
attacker = BIM(eps=0.15, eps_iter=0.01, n_iter=50, clip_max=-0.5, clip_min=0.5)

for i, (img, label) in enumerate(test_loader):
    if i == 2:
        break
    else:
        img_real = Variable(img.to(device))
        label = Variable(label.to(device))
        img_fake = attacker.generate(f,img_real,label)
        y_pred = f(img_fake)
        print(label)
        print(torch.max(y_pred, 1)[1])    
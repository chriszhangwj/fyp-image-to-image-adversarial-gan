import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import target_models
import os
import torch
import target_models
import torch.optim as optim
import torch.nn as nn
from tensorflow.examples.tutorials.mnist import input_data
from torch.autograd import Variable
from prepare_dataset import load_dataset
from test_function import test_semitargeted
from generators import Generator_MNIST as Generator
from discriminators import Discriminator_MNIST as Discriminator
from torch.optim.lr_scheduler import StepLR

def get_feature(f, train_loader, device):
    feature_vec = np.array([]).reshape(0,64)
    #print(np.shape(feature_vec))
    for i, (img, label) in enumerate(train_loader):
        img = Variable(img.to(device))
        _, feature  = f(img)
        feature = feature.cpu()
        feature = feature.detach().numpy()
        #print(feature.size())
        if i%1000 == 0:
            print(i)
            #print(feature)
            #print(np.shape(feature))
            #print(feature_vec)
        feature_vec = np.vstack([feature_vec,feature])
    return feature_vec

def get_mnist(train_loader):
    image_vec = np.array([]).reshape(0,784)
    label_vec = np.array([]).reshape(0,1)
    for i, (img, label) in enumerate(train_loader):
        img = img.view(1,-1)
        label = label.view(1,-1)
        image_vec = np.vstack([image_vec,img])
        label_vec = np.vstack([label_vec,label])
        if i%1000 == 0:
            print(i)
    return image_vec, label_vec

def CWLoss(logits, target, num_classes=10, kappa=0):
    target_one_hot = []
    for x in target.cpu().numpy():
        temp = torch.eye(num_classes).type(logits.type())[x]
        temp = temp.cpu()
        temp = temp.numpy()
        target_one_hot.append(temp)       
    target_one_hot = torch.FloatTensor(target_one_hot).to(device)
    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)
    return torch.sum(torch.max(other-real, kappa))

# load target model
#device = 'cuda'
#model_name = 'Model_C'
#f = getattr(target_models, model_name)(1, 10)
#checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist_fc64.pth.tar'%(model_name))
#checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
#f.load_state_dict(checkpoint_f["state_dict"])
#f.eval()
#f.cuda()

# get features for training set
train_data, test_data, in_channels, num_classes = load_dataset('mnist')
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4) # do not shuffle data
#feature = get_feature(f,train_loader,device) # feature is float64, (60000,dim_feature)
#np.savetxt("feature_200_mnist.csv", feature, delimiter=",")
feature = np.genfromtxt("feature_200_mnist.csv", delimiter=',')
feature = feature[0:10000,:]
#print(np.shape(feature)) # (60000,dim_feature)
#print(type(feature)) # <class 'numpy.ndarray'>

#print(len(train_data)) # 60000

#image_vec, label_vec = get_mnist(train_loader)
#np.savetxt("mnist_train_image_numpy.csv", image_vec, delimiter=",")
#np.savetxt("mnist_train_label_numpy.csv", label_vec, delimiter=",")
#print(np.shape(train_data))

#image_vec = np.genfromtxt("mnist_train_image_numpy.csv", delimiter=',')
label_vec = np.genfromtxt("mnist_train_label_numpy.csv", delimiter=',')
label_vec = label_vec[0:10000]
#print(np.shape(feature))
#print(np.shape(label_vec))
# print(len(feature))

# different-class nearest-neighbour search
#j_idx_dict = np.zeros((10000,1))
#for i in range(len(feature)): # i for target image
#    print('searching for index %d'%(i))
#    i_feature = feature[i,:]
#    norm_best = 1e5
#    j_best = 0
#    for j in range(len(feature)):
#        if label_vec[j] != label_vec[i]: # target sample label not equal to current sample label
#            j_feature = feature[j,:]
#            # compute L2 norm 
#            norm_temp = np.linalg.norm(i_feature-j_feature)
#            #print(norm_temp)
#            if norm_temp < norm_best:
#                j_best = j # update best index so far
#                norm_best = norm_temp # update smallest norm so far
#    j_idx_dict[i] = j_best
#
#j_label_dict = np.zeros((10000,1))
#for i in range(len(j_idx_dict)):
#    j_label_idx = int(j_idx_dict[i].item()) # get index of current j
#    j_label_dict[i] = label_vec[j_label_idx] # get the label of current j
#    print(label_vec[i])
#    print(label_vec[j_label_idx] )

#np.savetxt("mnist_j_idx_dict_1e4.csv", j_idx_dict, delimiter=",")
j_idx_dict = np.genfromtxt("mnist_j_idx_dict_1e4.csv", delimiter=',')
#np.savetxt("mnist_j_label_dict_1e4.csv", j_label_dict, delimiter=",")
j_label_dict = np.genfromtxt("mnist_j_label_dict_1e4.csv", delimiter=',')

# prepare data for GAN training
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4)

def train_semitargeted(G, D, f, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps
    stop_load = False
    G.train()
    D.train()
    for i, (img, label) in enumerate(train_loader): 
        if stop_load == True:
            break
        else:
            valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
            fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)
            
            img_real = Variable(img.to(device))
            optimizer_G.zero_grad()

            pert = torch.clamp(G(img_real), -thres, thres) # clip to [-0.3,0.3]
            img_fake = pert + img_real
            img_fake = img_fake.clamp(min=0, max=1) # clip to [-1,1]

            y_pred = f(img_fake)
            #print(y_pred.size())
        
            # determine the corresponding target label
            y_target=[]
            idx_begin= i*128
            #print(idx_begin)
            #print(label)
            for j in range(128):
                idx_current = idx_begin+j
                idx_end = idx_begin+127
                #print(idx_end)
                if idx_end < 10000:
                    y_target.append(j_label_dict[idx_current])
                    #print(len(y_target))
                else:
                    stop_load = True
                    for k in range(10000-idx_begin):
                        y_target.append(j_label_dict[idx_current+k])
                    #print(10000-idx_begin)
                    #print(len(y_target))
                    y_pred = y_pred.cpu()
                    y_pred = y_pred.data.numpy()
                    temp = y_pred[:10000-idx_begin,:]
                    y_pred = torch.from_numpy(temp).float().to(device)
                    break
                
            y_target = torch.LongTensor(y_target).to(device)
                
            # Train the Generator
            # adversarial loss
            #y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            loss_adv = criterion_adv(y_pred, y_target)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
              
            # GAN Generator loss
            loss_gan = criterion_gan(D(img_fake), valid)
            # perturbation loss
            loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) - thres))
            # total generator loss
            loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge 
            loss_g.backward(torch.ones_like(loss_g))
            optimizer_G.step()

            optimizer_D.zero_grad()
            if i % num_steps == 0:
                # Train the Discriminator
                loss_real = criterion_gan(D(img_real), valid)
                loss_fake = criterion_gan(D(img_fake.detach()), fake)
                loss_d = 0.5*loss_real + 0.5*loss_fake
                loss_d.backward(torch.ones_like(loss_d))
                optimizer_D.step()
            n += y_target.size(0)
    #print(n)  
    return acc/n 

gpu = True
device = 'cuda'
model_name = 'Model_C'
lr = 0.001
epochs = 100
    
D = Discriminator()
G = Generator()
t = getattr(target_models, model_name)(in_channels, num_classes)

checkpoint_path = os.path.join('saved', 'target_models', 'semitargeted', 'best_%s_mnist.pth.tar'%(model_name))
checkpoint = torch.load(checkpoint_path, map_location='cpu')
t.load_state_dict(checkpoint["state_dict"])
t.eval()

if gpu:
    D.cuda()
    G.cuda()
    t.cuda()
    
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

scheduler_G = StepLR(optimizer_G, step_size=30, gamma=0.1) # original step_size=5
scheduler_D = StepLR(optimizer_D, step_size=30, gamma=0.1)

criterion_adv = CWLoss # loss for fooling target model
criterion_gan = nn.MSELoss() # for gan loss
alpha = 1 # gan loss multiplication factor
beta = 1 # for hinge loss
num_steps = 3 # number of generator updates for 1 discriminator update
thres = c = 0.3 # perturbation bound, used in loss_hinge

device = 'cuda' if gpu else 'cpu'

acc_train_epoch = np.array([]).reshape(0,1)
acc_test_epoch = np.array([]).reshape(0,1)


for epoch in range(epochs):
    acc_train = train_semitargeted(G, D, t, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps, verbose=True)
    acc_test, _ = test_semitargeted(G, t, thres, test_loader, epoch, epochs, device, verbose=True)

    scheduler_G.step()
    scheduler_D.step()

    print("     "*20, end="\r")
    print('Epoch [%d/%d]\t\t\t'%(epoch+1, epochs))
    print('Train Acc: %.5f'%(acc_train))
    print('Test Acc: %.5f'%(acc_test))
    print('\n')

    torch.save({"epoch": epoch+1,
                "epochs": epochs,
                "thres": thres,
                "state_dict": G.state_dict(),
                "acc_test": acc_test,
                "optimizer": optimizer_G.state_dict()
                }, "saved/target_models/semitargeted/generators/bound_%.1f/%s_%s.pth.tar"%(thres, model_name, 'semitargeted'))

    acc_train_epoch=np.vstack([acc_train_epoch, acc_train])
    acc_test_epoch=np.vstack([acc_test_epoch, acc_test])
    
fig, ax = plt.subplots()    
ax.plot(acc_train_epoch, label='acc_train')
ax.plot(acc_test_epoch, label='acc_test')
ax.set(xlabel='Steps (Number of batches)', ylabel='Success rate',title='Attack success rate')
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major',linestyle='-')
ax.grid(which='minor',linestyle=':')
plt.legend(loc='upper right')
plt.show()
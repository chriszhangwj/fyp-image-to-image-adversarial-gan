import torch
import pytorch_ssim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
from noise import pnoise2
from utils import toZeroThreshold
torch.manual_seed(0)
#from operator import itemgetter

def train(G, D, f, target, is_targeted, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps

    G.train()
    D.train()
    
    for i, (img, label) in enumerate(train_loader):
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)

        img_real = Variable(img.to(device))

        optimizer_G.zero_grad()

        pert = torch.clamp(G(img_real), -thres, thres) # clip to [-0.3,0.3]
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1) # clip to [-1,1]

        y_pred = f(img_fake)
        # Train the Generator
        # adversarial loss
        if is_targeted:
            y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            
            loss_adv = criterion_adv(y_pred, y_target, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else:
            y_true = Variable(label.to(device))
            loss_adv = criterion_adv(y_pred, y_true, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        # GAN Generator loss
        loss_gan = criterion_gan(D(img_fake), valid)
        # perturbation loss
        loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) - thres))
        # total generator loss
        loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge
        # alternative loss functions
        #loss_g =  torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) + loss_adv # pert norm + adv loss
        #loss_g = loss_hinge + loss_adv # pert loss + adv loss
        
        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()

        optimizer_D.zero_grad()
        if i % num_steps == 0:
            # Train the Discriminator
            loss_real = criterion_gan(D(img_real), valid)
            loss_fake = criterion_gan(D(img_fake.detach()), fake)
            loss_d = 0.5*loss_real + 0.5*loss_fake # as defined in LSGAN paper, method 2
            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()

        n += img.size(0)
        #print("Epoch [%d/%d]: [%d/%d], D Loss: %1.4f, G Loss: %3.4f [H %3.4f, A %3.4f], Acc: %.4f"
        #    %(epoch+1, epochs, i, len(train_loader), loss_d.mean().item(), loss_g.mean().item(), loss_hinge.mean().item(), loss_adv.mean().item(), acc/n) , end="\r")
    return acc/n


def train_semitargeted(G, D, f, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps
    #target_pair = {'0':'5','1':'8','2':'8','3':'5','4':'9','5':'3','6':'2','7':'9','8':'5','9':'4'} # source-target pair dictionary
    #target_pair = {0:5,1:8,2:8,3:5,4:9,5:3,6:2,7:9,8:5,9:4} # source-target pair dictionary L2-norm
    target_pair = {0:5,1:8,2:3,3:5,4:9,5:8,6:2,7:9,8:5,9:4} # source-target pair dictionary SSIM
    
    G.train()
    D.train()
    for i, (img, label) in enumerate(train_loader): 
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)

        img_real = Variable(img.to(device))

        optimizer_G.zero_grad()

        pert = torch.clamp(G(img_real), -thres, thres) # clip to [-0.3,0.3]
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1) # clip to [-1,1]

        y_pred = f(img_fake)
        #print(type(y_pred))
        
        # determine the corresponding target label
        y_target=[]
        #print(label)
        for x in label.numpy():
            #target_pair.get(x)
             y_target.append(target_pair.get(x))
        y_target = torch.LongTensor(y_target).to(device)
        #print(y_target)
        
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
        # alternative loss functions as described in paper
        #loss_g =  torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) + loss_adv # pert norm + adv loss
        #loss_g = loss_hinge + loss_adv # pert loss + adv loss
        
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

        n += img.size(0)

        #print("Epoch [%d/%d]: [%d/%d], D Loss: %1.4f, G Loss: %3.4f [H %3.4f, A %3.4f], Acc: %.4f"
        #    %(epoch+1, epochs, i, len(train_loader), loss_d.mean().item(), loss_g.mean().item(), loss_hinge.mean().item(), loss_adv.mean().item(), acc/n) , end="\r")
    return acc/n


def train_plot(G, D, f, target, is_targeted, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps

    G.train()
    D.train()
    
    loss_adv_hist = np.array([]).reshape(0,1)
    loss_gan_hist = np.array([]).reshape(0,1)
    loss_hinge_hist = np.array([]).reshape(0,1)
    loss_g_hist = np.array([]).reshape(0,1)
    loss_d_hist = np.array([]).reshape(0,1)
    
    for i, (img, label) in enumerate(train_loader):
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)

        img_real = Variable(img.to(device))

        optimizer_G.zero_grad()

        pert = torch.clamp(G(img_real), -thres, thres) # clip to [-0.3,0.3]
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1) # clip to [-1,1]

        y_pred = f(img_fake)
        # Train the Generator
        # adversarial loss
        if is_targeted:
            y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            
            loss_adv = criterion_adv(y_pred, y_target, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else:
            y_true = Variable(label.to(device))
            loss_adv = criterion_adv(y_pred, y_true, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        # GAN Generator loss
        loss_gan = criterion_gan(D(img_fake), valid)

        # perturbation loss
        loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) - thres))

        # total generator loss
        loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge
        # alternative loss functions as described in paper
        #loss_g =  torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) + loss_adv # pert norm + adv loss
        #loss_g = loss_hinge + loss_adv # pert loss + adv loss
        
        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()

        optimizer_D.zero_grad()
        if i % num_steps == 0:
            # Train the Discriminator
            loss_real = criterion_gan(D(img_real), valid)
            loss_fake = criterion_gan(D(img_fake.detach()), fake)
            loss_d = 0.5*loss_real + 0.5*loss_fake # as defined in LSGAN paper
            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()

        n += img.size(0)
    
        loss_adv=loss_adv.cpu()
        loss_gan=loss_gan.cpu()
        loss_hinge=loss_hinge.cpu()
        loss_g = loss_g.cpu()
        loss_d = loss_d.cpu()
        loss_adv_hist=np.vstack([loss_adv_hist, loss_adv.detach().numpy()])
        loss_gan_hist=np.vstack([loss_gan_hist, loss_gan.detach().numpy()])
        loss_hinge_hist=np.vstack([loss_hinge_hist, loss_hinge.detach().numpy()])
        loss_g_hist=np.vstack([loss_g_hist, loss_g.detach().numpy()])
        loss_d_hist=np.vstack([loss_d_hist, loss_d.detach().numpy()])

        #print("Epoch [%d/%d]: [%d/%d], D Loss: %1.4f, G Loss: %3.4f [H %3.4f, A %3.4f], Acc: %.4f"
        #    %(epoch+1, epochs, i, len(train_loader), loss_d.mean().item(), loss_g.mean().item(), loss_hinge.mean().item(), loss_adv.mean().item(), acc/n) , end="\r")
    return acc/n,loss_adv_hist,loss_gan_hist,loss_hinge_hist, loss_g_hist, loss_d_hist

def train_plot_cifar10(G, D, f, target, is_targeted, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps

    G.train()
    D.train()
    
    loss_adv_hist = np.array([]).reshape(0,1)
    loss_gan_hist = np.array([]).reshape(0,1)
    loss_hinge_hist = np.array([]).reshape(0,1)
    loss_g_hist = np.array([]).reshape(0,1)
    loss_d_hist = np.array([]).reshape(0,1)
    
    for i, (img, label) in enumerate(train_loader):
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)

        img_real = Variable(img.to(device))

        optimizer_G.zero_grad()

        pert = torch.clamp(G(img_real), -thres, thres) # clip to [-0.3,0.3]
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=-1, max=1) # clip to [-1,1]

        y_pred = f(img_fake)
        # Train the Generator
        # adversarial loss
        if is_targeted:
            y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            
            loss_adv = criterion_adv(y_pred, y_target, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else:
            y_true = Variable(label.to(device))
            loss_adv = criterion_adv(y_pred, y_true, is_targeted)
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        # GAN Generator loss
        loss_gan = criterion_gan(D(img_fake), valid)

        # perturbation loss
        loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) - thres))

        # total generator loss
        loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge
        # alternative loss functions as described in paper
        #loss_g =  torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) + loss_adv # pert norm + adv loss
        #loss_g = loss_hinge + loss_adv # pert loss + adv loss
        
        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()

        optimizer_D.zero_grad()
        if i % num_steps == 0:
            # Train the Discriminator
            loss_real = criterion_gan(D(img_real), valid*0.8)
            loss_fake = criterion_gan(D(img_fake.detach()), fake)
            loss_d = 0.5*loss_real + 0.5*loss_fake 
            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()

        n += img.size(0)
    
        loss_adv=loss_adv.cpu()
        loss_gan=loss_gan.cpu()
        loss_hinge=loss_hinge.cpu()
        loss_g = loss_g.cpu()
        loss_d = loss_d.cpu()
        loss_adv_hist=np.vstack([loss_adv_hist, loss_adv.detach().numpy()])
        loss_gan_hist=np.vstack([loss_gan_hist, loss_gan.detach().numpy()])
        loss_hinge_hist=np.vstack([loss_hinge_hist, loss_hinge.detach().numpy()])
        loss_g_hist=np.vstack([loss_g_hist, loss_g.detach().numpy()])
        loss_d_hist=np.vstack([loss_d_hist, loss_d.detach().numpy()])
    return acc/n,loss_adv_hist,loss_gan_hist,loss_hinge_hist, loss_g_hist, loss_d_hist

def train_baseline(G, D, f, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    # only consider untargeted
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps

    G.train()
    D.train()
    
    loss_adv_hist = np.array([]).reshape(0,1)
    loss_gan_hist = np.array([]).reshape(0,1)
    loss_hinge_hist = np.array([]).reshape(0,1)
    loss_g_hist = np.array([]).reshape(0,1)
    loss_d_hist = np.array([]).reshape(0,1)
    
    for i, (img, label) in enumerate(train_loader):
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)
        img_real = Variable(img.to(device))
        optimizer_G.zero_grad()

        img_fake = torch.clamp(G(img_real), 0, 1) # clip to [0, 1]
        pert = img_fake - img_real # pert is for the entire batch
        y_pred = f(img_fake)
        
        # Train the Generator
        # adversarial loss
        y_true = Variable(label.to(device))
        loss_adv = criterion_adv(y_pred, y_true, is_targeted=False)
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        # GAN Generator loss
        loss_gan = criterion_gan(D(img_fake), valid)
        # perturbation loss
        loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) - thres))
        # total generator loss
        loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge
        # alternative loss functions
        #loss_g =  torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) + loss_adv # pert norm + adv loss
        #loss_g = loss_hinge + loss_adv # pert loss + adv loss
        
        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()

        optimizer_D.zero_grad()
        if i % num_steps == 0:
            print('update D')
            # Train the Discriminator
            #loss_real = criterion_gan(D(img_real), valid)
            loss_real = criterion_gan(D(img_real), valid*0.9) #label-smoothing
            loss_fake = criterion_gan(D(img_fake.detach()), fake)
            loss_d = 0.5*loss_real + 0.5*loss_fake # as defined in LSGAN paper, method 2
            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()

        n += img_real.size(0)
        
        loss_adv=loss_adv.cpu()
        loss_gan=loss_gan.cpu()
        loss_hinge=loss_hinge.cpu()
        loss_g = loss_g.cpu()
        loss_d = loss_d.cpu()
        loss_adv_hist=np.vstack([loss_adv_hist, loss_adv.detach().numpy()])
        loss_gan_hist=np.vstack([loss_gan_hist, loss_gan.detach().numpy()])
        loss_hinge_hist=np.vstack([loss_hinge_hist, loss_hinge.detach().numpy()])
        loss_g_hist=np.vstack([loss_g_hist, loss_g.detach().numpy()])
        loss_d_hist=np.vstack([loss_d_hist, loss_d.detach().numpy()])
        
    img = img.squeeze()
    plt.figure(figsize=(2,2))
    plt.imshow(img[1,:,:], cmap = 'gray')
    plt.title('Real image without noise: digit %d'%(label[1]))
    plt.show()    
    
    img_real = img_real.cpu()
    img_real = img_real.data.squeeze().numpy()
    plt.figure(figsize=(2,2))
    plt.imshow(img_real[1,:,:], cmap = 'gray')
    plt.title('Real image with noise: digit %d'%(label[1]))
    plt.show()    
            
    return acc/n,loss_adv_hist,loss_gan_hist,loss_hinge_hist, loss_g_hist, loss_d_hist

def train_perlin(G, D, f, M, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    # only consider untargeted
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps

    G.train()
    D.train()
    
    loss_adv_hist = np.array([]).reshape(0,1)
    loss_gan_hist = np.array([]).reshape(0,1)
    loss_hinge_hist = np.array([]).reshape(0,1)
    loss_g_hist = np.array([]).reshape(0,1)
    loss_d_hist = np.array([]).reshape(0,1)
    loss_real_hist = np.array([]).reshape(0,1)
    loss_fake_hist = np.array([]).reshape(0,1)

    noise = perlin(size = 28, period = 60, octave = 1, freq_sine = 36) # [0,1]
    noise = (noise - 0.5)*2 # [-1,1]
    noise = M * noise.squeeze()
    
    for i, (img, label) in enumerate(train_loader): 
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)
        img = img.cpu()
        img = img.detach().numpy()
        img_real = 255 * img
        img_noise = np.tile(noise,(img_real.shape[0],1,1,1))
        img_real = img_real + img_noise
        img_real = img_real/255.0
        img_real = Variable(torch.from_numpy(img_real).to(device))

        optimizer_G.zero_grad()

        img_fake = torch.clamp(G(img_real), 0, 1) # clip to [0, 1]
        pert = img_fake - img_real # pert is for the entire batch
        y_pred = f(img_fake)
        
        # Train the Generator
        # adversarial loss
        y_true = Variable(label.to(device))
        loss_adv = criterion_adv(y_pred, y_true, is_targeted=False)
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        # GAN Generator loss
        loss_gan = criterion_gan(D(img_fake), valid) # loss is low when the discriminator think it's valid; loss is high when discriminator tells it's fake
        # perturbation loss
        loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=1, dim=1)))
        # total generator loss
        loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge
        # alternative loss functions
        #loss_g =  torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) + loss_adv # pert norm + adv loss
        #loss_g = loss_hinge + loss_adv # pert loss + adv loss
        
        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()

        optimizer_D.zero_grad()
        if i % num_steps == 0:
            print('update D')
#            if use_noise == True:
#                wgn = Variable(img_real.data.new(img_real.size()).normal_(0,0.1))
            
            # Train the Discriminator
            # loss_real = criterion_gan(D(img_real), valid)
            loss_real = criterion_gan(D(img_real), valid*0.5)
            loss_fake = criterion_gan(D(img_fake.detach()), fake)
            loss_d = loss_real + loss_fake 
            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()
            loss_real = loss_real.cpu()
            loss_real = loss_real.data.squeeze().numpy()
            loss_fake = loss_fake.cpu()
            loss_fake = loss_fake.data.squeeze().numpy()
            loss_real_hist=np.vstack([loss_real_hist, loss_real])
            loss_fake_hist=np.vstack([loss_fake_hist, loss_fake])
            
        n += img_real.size(0)
        
        loss_adv=loss_adv.cpu()
        loss_gan=loss_gan.cpu()
        loss_hinge=loss_hinge.cpu()
        loss_g = loss_g.cpu()
        loss_d = loss_d.cpu()
        loss_adv_hist=np.vstack([loss_adv_hist, loss_adv.detach().numpy()])
        loss_gan_hist=np.vstack([loss_gan_hist, loss_gan.detach().numpy()])
        loss_hinge_hist=np.vstack([loss_hinge_hist, loss_hinge.detach().numpy()])
        loss_g_hist=np.vstack([loss_g_hist, loss_g.detach().numpy()])
        loss_d_hist=np.vstack([loss_d_hist, loss_d.detach().numpy()])
        
#    img = img.squeeze()
#    plt.figure(figsize=(1.5,1.5))
#    plt.imshow(img[1,:,:], cmap = 'gray')
#    plt.title('Real image without noise: digit %d'%(label[1]))
#    plt.show()    
#    
#    img_real = img_real.cpu()
#    img_real = img_real.data.squeeze().numpy()
#    plt.figure(figsize=(1.5,1.5))
#    plt.imshow(img_real[1,:,:], cmap = 'gray')
#    plt.title('Real image with noise: digit %d'%(label[1]))
#    plt.show()    
            
    return acc/n,loss_adv_hist,loss_gan_hist,loss_hinge_hist, loss_g_hist, loss_d_hist, loss_real_hist, loss_fake_hist


def train_baseline_ACGAN(G, D, f, criterion_adv, criterion_gan, criterion_aux, alpha, beta, gamma, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps
    G.train()
    D.train()
    loss_adv_hist = np.array([]).reshape(0,1)
    loss_gan_hist = np.array([]).reshape(0,1)
    loss_hinge_hist = np.array([]).reshape(0,1)
    loss_g_hist = np.array([]).reshape(0,1)
    loss_d_hist = np.array([]).reshape(0,1)
    loss_real_hist = np.array([]).reshape(0,1)
    loss_fake_hist = np.array([]).reshape(0,1)
    loss_aux_hist = np.array([]).reshape(0,1)
    
    for i, (img, label) in enumerate(train_loader):
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)
        img_real = Variable(img.to(device))
        optimizer_G.zero_grad()
        
        img_fake = torch.clamp(G(img_real), 0, 1)

        #img_fake = torch.clamp(G(img_real), 0, 1) # clip to [0, 1]
        pert = img_fake - img_real # pert is for the entire batch
        y_pred = f(img_fake)
        # Train the Generator
        # adversarial loss
        y_true = Variable(label.to(device))
        loss_adv = criterion_adv(y_pred, y_true, is_targeted=False)
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        # GAN Generator loss
        x1_fake, _ = D(img_fake)
        loss_gan = criterion_gan(x1_fake, valid)
        # perturbation loss
        loss_hinge_1 = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=1, dim=1)))
        
        loss_hinge_2 = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1)))

        loss_hinge = 0.2 * loss_hinge_1 + 0.8 * loss_hinge_2
        
        # total generator loss
        loss_g = 0.2 * loss_adv + alpha*loss_gan + beta*loss_hinge            
        
        # total generator loss
        #loss_g = 0.1*loss_adv + alpha*loss_gan + beta*loss_hinge      
        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()
        optimizer_D.zero_grad()
        
        if i % num_steps == 0:
            #print('update D')
            # Train the Discriminator
            #loss_real = criterion_gan(D(img_real), valid*0.9) #label-smoothing
            #loss_real = criterion_gan(D(img_real), valid*0.5)
            #loss_fake = criterion_gan(D(img_fake.detach()), fake)
            
            x1_real, _ = D(img_real)
            x1_fake, x2_fake = D(img_fake.detach())
            loss_real = criterion_gan(x1_real, valid*0.6)
            loss_fake = criterion_gan(x1_fake, fake)
            loss_aux = criterion_aux(x2_fake, y_true)
            loss_d = 0.5*loss_real + 0.5*loss_fake + gamma * loss_aux
            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()

        n += img_real.size(0)
        
        loss_adv=loss_adv.cpu()
        loss_gan=loss_gan.cpu()
        loss_hinge=loss_hinge.cpu()
        loss_g = loss_g.cpu()
        loss_d = loss_d.cpu()
        loss_real = loss_real.cpu()
        loss_fake = loss_fake.cpu()
        loss_aux = loss_aux.cpu()
        loss_adv_hist=np.vstack([loss_adv_hist, loss_adv.detach().numpy()])
        loss_gan_hist=np.vstack([loss_gan_hist, loss_gan.detach().numpy()])
        loss_hinge_hist=np.vstack([loss_hinge_hist, loss_hinge.detach().numpy()])
        loss_g_hist=np.vstack([loss_g_hist, loss_g.detach().numpy()])
        loss_d_hist=np.vstack([loss_d_hist, loss_d.detach().numpy()])
        
        loss_real_hist=np.vstack([loss_real_hist, loss_real.detach().numpy()])
        loss_fake_hist=np.vstack([loss_fake_hist, loss_fake.detach().numpy()])
        loss_aux_hist=np.vstack([loss_aux_hist, loss_aux.detach().numpy()])
            
    return acc/n,loss_adv_hist,loss_gan_hist,loss_hinge_hist, loss_g_hist, loss_d_hist, loss_real_hist, loss_fake_hist, loss_aux_hist

def train_baseline_CIFAR10(G, D, f, criterion_adv, criterion_gan, criterion_aux, alpha, beta, gamma, lam, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps
    G.train()
    D.train()
    loss_adv_hist = np.array([]).reshape(0,1)
    loss_gan_hist = np.array([]).reshape(0,1)
    loss_hinge_hist = np.array([]).reshape(0,1)
    loss_g_hist = np.array([]).reshape(0,1)
    loss_d_hist = np.array([]).reshape(0,1)
    loss_real_hist = np.array([]).reshape(0,1)
    loss_fake_hist = np.array([]).reshape(0,1)
    loss_aux_hist = np.array([]).reshape(0,1)
    
    for i, (img, label) in enumerate(train_loader):
        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)
        img_real = Variable(img.to(device))
        optimizer_G.zero_grad()
        
        img_fake = torch.clamp(G(img_real), -1, 1)

        pert = img_fake - img_real # pert is for the entire batch
        y_pred = f(img_fake)
        # Train the Generator
        # adversarial loss
        y_true = Variable(label.to(device))
        loss_adv = criterion_adv(y_pred, y_true, is_targeted=False)
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        # GAN Generator loss
        x1_fake, _ = D(img_fake)
        loss_gan = criterion_gan(x1_fake, valid)
        # perturbation loss
        loss_hinge_1 = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=1, dim=1)))
        
        loss_hinge_2 = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1)))

        loss_hinge = 1 * loss_hinge_1 + 0 * loss_hinge_2
        
        # total generator loss
        loss_g = beta * loss_adv + alpha*loss_gan + gamma*loss_hinge            
        
        # total generator loss
        #loss_g = 0.1*loss_adv + alpha*loss_gan + beta*loss_hinge      
        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()
        optimizer_D.zero_grad()
        
        if i % num_steps == 0:
            #print('update D')
            # Train the Discriminator
            #loss_real = criterion_gan(D(img_real), valid*0.9) #label-smoothing
            #loss_real = criterion_gan(D(img_real), valid*0.5)
            #loss_fake = criterion_gan(D(img_fake.detach()), fake)
            
            x1_real, _ = D(img_real)
            x1_fake, x2_fake = D(img_fake.detach())
            loss_real = criterion_gan(x1_real, valid*0.8)
            loss_fake = criterion_gan(x1_fake, fake)
            loss_aux = criterion_aux(x2_fake, y_true)
            loss_d = 0.5*loss_real + 0.5*loss_fake + lam * loss_aux
            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()

        n += img_real.size(0)
        
        loss_adv=loss_adv.cpu()
        loss_gan=loss_gan.cpu()
        loss_hinge=loss_hinge.cpu()
        loss_g = loss_g.cpu()
        loss_d = loss_d.cpu()
        loss_real = loss_real.cpu()
        loss_fake = loss_fake.cpu()
        loss_aux = loss_aux.cpu()
        loss_adv_hist=np.vstack([loss_adv_hist, loss_adv.detach().numpy()])
        loss_gan_hist=np.vstack([loss_gan_hist, loss_gan.detach().numpy()])
        loss_hinge_hist=np.vstack([loss_hinge_hist, loss_hinge.detach().numpy()])
        loss_g_hist=np.vstack([loss_g_hist, loss_g.detach().numpy()])
        loss_d_hist=np.vstack([loss_d_hist, loss_d.detach().numpy()])
        
        loss_real_hist=np.vstack([loss_real_hist, loss_real.detach().numpy()])
        loss_fake_hist=np.vstack([loss_fake_hist, loss_fake.detach().numpy()])
        loss_aux_hist=np.vstack([loss_aux_hist, loss_aux.detach().numpy()])
            
    return acc/n,loss_adv_hist,loss_gan_hist,loss_hinge_hist, loss_g_hist, loss_d_hist, loss_real_hist, loss_fake_hist, loss_aux_hist

def train_patchgan_CIFAR10(G, D, f, criterion_adv, criterion_gan, criterion_aux, alpha, beta, gamma, lam, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
    n = 0
    acc = 0 # attack success rate
    num_steps = num_steps
    G.train()
    D.train()
    loss_adv_hist = np.array([]).reshape(0,1)
    loss_gan_hist = np.array([]).reshape(0,1)
    loss_hinge_hist = np.array([]).reshape(0,1)
    loss_g_hist = np.array([]).reshape(0,1)
    loss_d_hist = np.array([]).reshape(0,1)
    loss_real_hist = np.array([]).reshape(0,1)
    #loss_fake_hist = np.array([]).reshape(0,1)
    
    for i, (img, label) in enumerate(train_loader):
        img_real = Variable(img.to(device))
        for param in D.parameters():
             param.requires_grad = False
        optimizer_G.zero_grad()

        img_fake = torch.clamp(G(img_real), -1, 1)

        pert = img_fake - img_real # pert is for the entire batch
        y_pred = f(img_fake)
        # Train the Generator
        # adversarial loss
        y_true = Variable(label.to(device))
        loss_adv = criterion_adv(y_pred, y_true, is_targeted=False)
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()

        # GAN Generator loss
        loss_fake = D(img_fake)
        loss_gan = criterion_gan(loss_fake, True)
        # perturbation loss
        loss_hinge_1 = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=1, dim=1)))
        
        loss_hinge_2 = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1)))

        loss_hinge = 1 * loss_hinge_1 + 0 * loss_hinge_2
        
        # total generator loss
        loss_g = beta * loss_adv + alpha*loss_gan + gamma*loss_hinge            
        
        # total generator loss
        #loss_g = 0.1*loss_adv + alpha*loss_gan + beta*loss_hinge      
        loss_g.backward(torch.ones_like(loss_g))
        optimizer_G.step()
        
        if i % num_steps == 0:
            for param in D.parameters():
                param.requires_grad = True
            optimizer_D.zero_grad()
            #print('update D')
            # Train the Discriminator
            #loss_real = criterion_gan(D(img_real), valid*0.9) #label-smoothing
            #loss_real = criterion_gan(D(img_real), valid*0.5)
            #loss_fake = criterion_gan(D(img_fake.detach()), fake)
            
            pred_real = D(img_real)
            pred_fake = D(img_fake.detach())
            loss_real = criterion_gan(pred_real, True)
            print(loss_real)
            loss_fake = criterion_gan(pred_fake, False)
            print(loss_fake)
            loss_d = 0.5*loss_real + 0.5*loss_fake
            loss_d.backward(torch.ones_like(loss_d))
            optimizer_D.step()

        n += img_real.size(0)
        
        loss_adv=loss_adv.cpu()
        loss_gan=loss_gan.cpu()
        loss_hinge=loss_hinge.cpu()
        loss_g = loss_g.cpu()
        loss_d = loss_d.cpu()
        loss_real = loss_real.cpu()
        #loss_fake = loss_fake.cpu()
        loss_adv_hist=np.vstack([loss_adv_hist, loss_adv.detach().numpy()])
        loss_gan_hist=np.vstack([loss_gan_hist, loss_gan.detach().numpy()])
        loss_hinge_hist=np.vstack([loss_hinge_hist, loss_hinge.detach().numpy()])
        loss_g_hist=np.vstack([loss_g_hist, loss_g.detach().numpy()])
        loss_d_hist=np.vstack([loss_d_hist, loss_d.detach().numpy()])
        loss_real_hist=np.vstack([loss_real_hist, loss_real.detach().numpy()])
        #loss_fake_hist=np.vstack([loss_fake_hist, loss_fake.detach().numpy()])
            
    return acc/n,loss_adv_hist,loss_gan_hist,loss_hinge_hist, loss_g_hist, loss_d_hist, loss_real_hist

import torch
import pytorch_ssim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
from noise import pnoise2
from utils import perlin, colorize
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
            loss_real = criterion_gan(D(img_real), valid)
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

def train_perlin(G, D, f, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
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

    M = 50
    noise = perlin(size = 28, period = 60, octave = 1, freq_sine = 36) # [0,1]
    noise = (noise - 0.5)*2 # [-1,1]
    #payload = (np.sign(noise.reshape(28, 28, 1)) + 1) / 2 # [-1,1] binary # [0,2] binary # [0,1] binary
    noise = M * noise.squeeze()
    #payload = M * (payload-0.5)*2 # [-M,M] binary
    
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
            loss_real = criterion_gan(D(img_real), valid)
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







#def train_baseline_BEGAN(G, D, f, thres, criterion_adv, criterion_gan, alpha, beta, train_loader, optimizer_G, optimizer_D, epoch, epochs, device, num_steps=3, verbose=True):
#    # only consider untargeted
#    n = 0
#    acc = 0 # attack success rate
#    num_steps = num_steps
#
#    G.train()
#    D.train()
#    
#    loss_adv_hist = np.array([]).reshape(0,1)
#    loss_gan_hist = np.array([]).reshape(0,1)
#    loss_hinge_hist = np.array([]).reshape(0,1)
#    loss_g_hist = np.array([]).reshape(0,1)
#    loss_d_hist = np.array([]).reshape(0,1)
#    
#    for i, (img, label) in enumerate(train_loader):
#        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0).to(device), requires_grad=False)
#        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0).to(device), requires_grad=False)
#
#        img_real = Variable(img.to(device))
#
#        optimizer_G.zero_grad()
#
#        img_fake = torch.clamp(G(img_real), 0, 1) # clip to [0, 1]
#        pert = img_fake - img_real 
#        y_pred = f(img_fake)
        
#        
#        # Train the Discriminiator
#                if i % num_steps == 0:
#            print('update D')
#            # Train the Discriminator
#            loss_real = criterion_gan(D(img_real), valid)
#            loss_fake = criterion_gan(D(img_fake.detach()), fake)
#            loss_d = 0.5*loss_real + 0.5*loss_fake # as defined in LSGAN paper, method 2
#            loss_d.backward(torch.ones_like(loss_d))
#            optimizer_D.step()
        
        
        
        
        
        # Train the Generator
        # adversarial loss
#        y_true = Variable(label.to(device))
#        loss_adv = criterion_adv(y_pred, y_true, is_targeted=False)
#        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item()
#
#        # GAN Generator loss
#        loss_gan = criterion_gan(D(img_fake), valid)
#        # perturbation loss
#        loss_hinge = torch.mean(torch.max(torch.zeros(1, ).type(y_pred.type()), torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) - thres))
#        # total generator loss
#        loss_g = loss_adv + alpha*loss_gan + beta*loss_hinge
#        # alternative loss functions
#        #loss_g =  torch.norm(pert.view(pert.size(0), -1), p=2, dim=1) + loss_adv # pert norm + adv loss
#        #loss_g = loss_hinge + loss_adv # pert loss + adv loss
#        
#        loss_g.backward(torch.ones_like(loss_g))
#        optimizer_G.step()
#
#        optimizer_D.zero_grad()
#
#
#        n += img.size(0)
#        
#        loss_adv=loss_adv.cpu()
#        loss_gan=loss_gan.cpu()
#        loss_hinge=loss_hinge.cpu()
#        loss_g = loss_g.cpu()
#        loss_d = loss_d.cpu()
#        loss_adv_hist=np.vstack([loss_adv_hist, loss_adv.detach().numpy()])
#        loss_gan_hist=np.vstack([loss_gan_hist, loss_gan.detach().numpy()])
#        loss_hinge_hist=np.vstack([loss_hinge_hist, loss_hinge.detach().numpy()])
#        loss_g_hist=np.vstack([loss_g_hist, loss_g.detach().numpy()])
#        loss_d_hist=np.vstack([loss_d_hist, loss_d.detach().numpy()])
#    return acc/n,loss_adv_hist,loss_gan_hist,loss_hinge_hist, loss_g_hist, loss_d_hist


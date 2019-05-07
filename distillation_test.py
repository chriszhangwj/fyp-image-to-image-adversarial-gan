"""
 Idea from : Distilling the Knowledge in a Neural Network
 https://arxiv.org/pdf/1503.02531.pdf
"""
import os
import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchsummary import summary
from torch.autograd import Variable

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from prepare_dataset import load_dataset

dataset_name = 'mnist'
batch_size=32
num_workers=4
train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class FCNetwork(nn.Module):
    def __init__(self, hidden_size=1200, T=1, d=.5):
        super(FCNetwork, self).__init__()
        self.T = T
        self.clf = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(d),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 10)
        )
        self.softmax = nn.Softmax() # defines softmax function

    def forward(self, x):
        out = self.clf(x) # obtain output from defined network model
        out = out / self.T # scale output with temperature parameter
        return self.softmax(out) # pass scaled output through softmax

def train(model, optimizer, loss_function, nb_epochs=30):
    for i in trange(nb_epochs):
        big_network.train()
        loss_history = []
        for x, target in train_loader:
            x, target = Variable(x).view([batch_size, -1]), Variable(target)
            out = model(x)
            loss = loss_function(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        print(sum(loss_history) / len(loss_history), end='\t')
        big_network.eval()
        loss_history = []
        for x, target in train_loader:
            x, target = Variable(x).view([batch_size, -1]), Variable(target)
            out = model(x)
            loss = loss_function(out, target)
            loss_history.append(loss.item())
        print(sum(loss_history) / len(loss_history))

def accuracy(model):
    model.eval()
    t = []
    for x, target in train_loader:
        x, target = Variable(x).view([batch_size, -1]), Variable(target)
        out = model(x)
        t += [sum(out.argmax(1) == target).item() / batch_size]
    return np.array(t).mean()

def distillation_loss_function(model_pred, model_pred_T, teach_pred_T, target, T, alpha=0.9): # compute the KD loss; hyperparameters are T and alpha
    return nn.KLDivLoss()(model_pred_T, teach_pred_T) * (T * T * alpha) +\
        nn.CrossEntropyLoss()(model_pred, target) * (1 - alpha)


def distill(student, teacher, T, optimizer, nb_epochs=30, alpha=0.9): # input arguments: distill model, blackbox model and temperature 
    student.T = T
    teacher.T = T
    # explicityly sepcify intented use of models
    student.train() # use student model to train
    teacher.eval() # use teacher model for evaluation
    print(alpha)
    #for i in trange(nb_epochs):
    for i in range(nb_epochs):
        loss_history = []
        for x, target in train_loader:
            x, target = Variable(x).view([batch_size, -1]), Variable(target)
            student_pred = student(x)
            student.T = T
            teacher.T = T
            student_pred_T = student(x) # obtain student prediction
            teacher_pred_T = teacher(x).detach() # obtain teacher prediction  #to not requires grad
            loss = distillation_loss_function(student_pred, student_pred_T, teacher_pred_T, target, alpha)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() # update parameters
            loss_history.append(loss.item())
            student.T = 1. # set temperature back to 1
            teacher.T = 1. # set temperature back to 1
        print(sum(loss_history) / float(len(loss_history)))
    student.T = 1. # set temperature back to 1
    teacher.T = 1. # set temperature back to 1

# Target network
big_network = FCNetwork(800)
optimizer = optim.Adam(big_network.parameters())
loss_function = nn.CrossEntropyLoss()
train(big_network, optimizer, loss_function, 5) 
print("teacher accuracy : ", accuracy(big_network))

small_network = FCNetwork(30, d=0.1) # dropout rate 0.1
optimizer = optim.Adam(small_network.parameters())
loss_function = nn.CrossEntropyLoss()
train(small_network, optimizer, loss_function, 5) 
print("small net accuracy : ", accuracy(small_network)) 

small_network_d = FCNetwork(30, d=.01) # dropout rate 0.01
optimizer = optim.Adam(small_network_d.parameters())
distill(small_network_d, big_network, 70, optimizer, 5, 0.9) 
print("small net as student accuracy : ", accuracy(small_network_d)) 
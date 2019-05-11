import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from torch.autograd import Variable
from prepare_dataset import load_dataset
from target_models import Model_distill, Model_C, FCNetwork

dataset_name = 'mnist'
batch_size=32
num_workers=4
train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def test(model, test_loader, criterion):
    model.eval()
    n = 0
    test_loss, test_acc = 0.0, 0.0
    for i, (X, y) in enumerate(test_loader):
        X = Variable(X.float().cuda())
        y = Variable(y.long().cuda())
        out = model(X)
        _, y_pred = torch.max(out.data, 1)
        loss = criterion(out, y)
        test_loss += loss.item() * X.size(0)
        test_acc += torch.sum(y_pred == y.data).item()
        n += X.size(0)
    return test_loss/n, test_acc/n

def test_distill(model, train_loader, criterion):
    model.eval()
    n = 0
    test_loss, test_acc = 0.0, 0.0
    for i, (X, y) in enumerate(train_loader): # use train data to evaluate performance of distilled model
        X = Variable(X.float().cuda())
        y = Variable(y.long().cuda())
        out = model(X)
        _, y_pred = torch.max(out.data, 1)
        loss = criterion(out, y)
        test_loss += loss.item() * X.size(0)
        test_acc += torch.sum(y_pred == y.data).item()
        n += X.size(0)
    return test_loss/n, test_acc/n

def train(model, train_loader, criterion, optimizer, epoch, epochs):
    model.train()
    n = 0
    train_loss, train_acc = 0.0, 0.0
    for i, (X, y) in enumerate(train_loader):
        X = Variable(X.float().cuda())
        y = Variable(y.long().cuda())
        optimizer.zero_grad()
        out = model(X)
        _, y_pred = torch.max(out.data, 1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
        train_acc += torch.sum(y_pred == y.data).item()
        n += X.size(0)
    return train_loss/n, train_acc/n

def distillation_loss_function(model_pred, model_pred_T, teach_pred_T, target, T, alpha=0.9): # compute the KD loss; hyperparameters are T and alpha
    return nn.KLDivLoss()(model_pred_T, teach_pred_T) * (T * T * alpha) +\
        nn.CrossEntropyLoss()(model_pred, target) * (1 - alpha)           
        # compare predicted probabilities (T) to target probabilities (T) and compare predicted probabilities to labels
        # the original paper suggests the use of cross entropy

def distill(student, teacher, T, optimizer, epoch, epochs, alpha=0.9):
    student.T = T
    teacher.T = T
    student.train() # use student model to train
    teacher.eval() # use teacher model for evaluation
    n = 0
    train_loss, train_acc = 0.0, 0.0
    for i, (X, y) in enumerate(test_loader): # use test data to train (as suggested by the paper, use a disjoint set of data as we assume access to train data)
        X = Variable(X.float().cuda())
        y = Variable(y.long().cuda())
        student_pred = student(X)
        student.T = T
        teacher.T = T
        student_pred_T = student(X) # obtain student prediction
        teacher_pred_T = teacher(X).detach() # obtain teacher prediction  #to not requires grad
        loss = distillation_loss_function(student_pred, student_pred_T, teacher_pred_T, y, alpha)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() # update parameters        
        student.T = 1. # set temperature back to 1
        teacher.T = 1. # set temperature back to 1
        n += X.size(0)
    return train_loss/n, train_acc/n    

def save_checkpoint(state, checkpoint_name, best_name):
    torch.save(state, checkpoint_name)
    if state['is_best']==True:
        shutil.copyfile(checkpoint_name, best_name)

# Note: the target model is trained with training data, and is considered as a blackbox. The distill model is 
# trained with testing data because we assume no access to the data on which the target model is trained 
# (i.e. the training data). Hence, we can assume full access to the data we use to query the target model
# (i.e. the testing data), including their ground-truth labels

# Target network
target_network = Model_C(in_channels,num_classes)
target_network.cuda()
epochs=10
best_acc = 0 
for epoch in range(epochs):    
    #optimizer = optim.Adam(target_network.parameters())
    optimizer = optim.SGD(target_network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().cuda()
    loss_function = nn.CrossEntropyLoss()
    train_loss, train_acc = train(target_network, train_loader, criterion, optimizer, epoch, epochs) 
    test_loss, test_acc = test(target_network, test_loader, criterion) 
    print("  "*40)
    print("Target network")
    print('Epoch [%3d/%3d]'%(epoch+1, epochs))
    print('Train Acc: %.8f' %(train_acc))
    print('Test Acc:  %.8f' %(test_acc))
    print('-'*30)
    is_best = test_acc > best_acc
    best_acc = max(best_acc, test_acc) # save the best trained distill model
    save_checkpoint({"epoch": epoch,
                    "state_dict": target_network.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "is_best": is_best,
                    }, checkpoint_name="saved/target_models/distill/checkpoint_%d_Model_C_mnist.pth.tar"%(epoch),
                        best_name="saved/target_models/distill/best_Model_C_mnist.pth.tar")
    
#small_network = Model_distill(in_channels,num_classes)
#small_network.cuda()     
#for epoch in range(epochs):    
#    optimizer = optim.Adam(small_network.parameters())
#    criterion = nn.CrossEntropyLoss().cuda()
#    loss_function = nn.CrossEntropyLoss()
#    train_loss, train_acc = train(small_network, train_loader, criterion, optimizer, epoch, epochs) 
#    test_loss, test_acc = test(small_network, test_loader, criterion) 
#    print("  "*40)
#    print("Student network")
#    print('Epoch [%3d/%3d]'%(epoch+1, epochs))
#    print('Train Acc: %.8f' %(train_acc))
#    print('Test Acc:  %.8f' %(test_acc))
#    print('-'*30)    
    
# Distillation network    
distill_network = Model_distill(in_channels,num_classes)
#distill_network = FCNetwork(in_channels,num_classes) 
distill_network.cuda() 
epochs = 50  
best_acc = 0
for epoch in range(epochs):    
    optimizer = optim.Adam(distill_network.parameters()) # SGD gives worse training process
    criterion = nn.CrossEntropyLoss().cuda()
    loss_function = nn.CrossEntropyLoss()
    distill(distill_network, target_network, 10, optimizer, epoch, epochs, 0.1) # hyperparameters T and alpha # using alpha=1.0 we assume no target labels available
    print("  "*40)
    print("Distillation network")
    print('Epoch [%3d/%3d]'%(epoch+1, epochs))
    print('-'*30) 
    _, test_acc = test_distill(distill_network, train_loader, criterion) 
    print('Test Acc: %.8f' %(test_acc))
    
    is_best = test_acc > best_acc
    best_acc = max(best_acc, test_acc) # save the best trained distill model
    save_checkpoint({"epoch": epoch,
                    "state_dict": distill_network.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "is_best": is_best,
                    }, checkpoint_name="saved/target_models/distill/checkpoint_%d_distill_Model_C_mnist.pth.tar"%(epoch),
                        best_name="saved/target_models/distill/best_distill_Model_C_mnist.pth.tar")
print('Best Test Acc: %.8f'%(best_acc))
    
#    
    
    




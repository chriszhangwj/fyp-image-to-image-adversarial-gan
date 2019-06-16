'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import os
import argparse
import shutil

from resnet import ResNet18_CAM
from prepare_dataset import load_dataset
from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
args = parser.parse_args()

device = 'cuda'
dataset_name = 'cifar10'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr=0.001
batch_size=128
n_epoch=4
resume=True

# Data
print('Preparing data..')
train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('Building model..')
net = ResNet18_CAM()
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=130, gamma=0.1)

summary(net, (3,28,28))

if resume==True:
    print('Resuming from checkpoint')
    checkpoint_path = 'saved/cifar10/target_models/checkpoint_cifar10_cam.pth.tar'
    checkpoint=torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    print('starting best_acc: ', best_acc)
    print('starting epoch: ', start_epoch)

def save_checkpoint(state, checkpoint_name, best_name):
    torch.save(state, checkpoint_name)
    if state['is_best']==True:
        shutil.copyfile(checkpoint_name, best_name)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs,_ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        print('Train [%2d]: [%4d/%4d]\tLoss: %1.4f\tAcc: %1.8f'
        %(epoch+1, batch_idx+1, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total), end="\n")

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,_ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            print('Test [%2d]: [%4d/%4d]\tLoss: %1.4f\tAcc: %1.8f'
            %(epoch+1, batch_idx+1, len(test_loader), test_loss/(batch_idx+1), 100.*correct/total), end="\n")            

    # Save checkpoint.
    acc = 100.*correct/total
    is_best = acc > best_acc
    best_acc = max(best_acc, acc)
    save_checkpoint({"epoch": epoch,
                     "state_dict": net.state_dict(),
                     "best_acc": best_acc,
                     "optimizer": optimizer.state_dict(),
                     "is_best": is_best,
                     }, checkpoint_name="saved/cifar10/target_models/checkpoint_%s_cam.pth.tar"%(dataset_name),
                     best_name="saved/cifar10/target_models/best_%s_cam.pth.tar"%(dataset_name))
    
#    if acc > best_acc:
#        print('Saving..')
#        state = {
#            'net': net.state_dict(),
#            'acc': acc,
#            'epoch': epoch,
#        }
#        if not os.path.isdir('checkpoint'):
#            os.mkdir('checkpoint')
#        torch.save(state, 'saved/cifar10/target_models/best_resnet18_model.pth.tar')
#        best_acc = acc


for epoch in range(start_epoch, start_epoch+n_epoch+1):
    train(epoch)
    test(epoch)
    scheduler.step()

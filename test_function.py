import torch
from torch.autograd import Variable
import pytorch_ssim


def test(G, f, target, is_targeted, thres, test_loader, epoch, epochs, device, verbose=True):
    n = 0
    acc = 0
    ssim = 0

    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)

        if is_targeted: # if targeted
            y_target = Variable(torch.ones_like(label).fill_(target).to(device))
            acc += torch.sum(torch.max(y_pred, 1)[1] == y_target).item()
        else: # if untargeted
            y_true = Variable(label.to(device))
            acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img.size(0)
#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    return acc/n, ssim/n # returns attach success rate


#def test_semitargeted(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
#    n = 0
#    acc = 0
#
#    G.eval()
#    for i, (img, label) in enumerate(test_loader):
#        img_real = Variable(img.to(device))
#
#        pert = torch.clamp(G(img_real), -thres, thres)
#        img_fake = pert + img_real
#        img_fake = img_fake.clamp(min=0, max=1)
#
#        y_pred = f(img_fake)
#
#        y_true = Variable(label.to(device))
#        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
#
#        n += img.size(0)
#
##        if verbose:
##            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
#    return acc/n # returns attach success rate

def test_semitargeted(G, f, thres, test_loader, epoch, epochs, device, verbose=True):
    n = 0
    acc = 0
    ssim = 0

    G.eval()
    for i, (img, label) in enumerate(test_loader):
        img_real = Variable(img.to(device))

        pert = torch.clamp(G(img_real), -thres, thres)
        img_fake = pert + img_real
        img_fake = img_fake.clamp(min=0, max=1)

        y_pred = f(img_fake)

        y_true = Variable(label.to(device))
        acc += torch.sum(torch.max(y_pred, 1)[1] != y_true).item() # when the prediction is wrong
        ssim += pytorch_ssim.ssim(img_real, img_fake).item()
        n += img.size(0)
#        if verbose:
#            print('Test [%d/%d]: [%d/%d]' %(epoch+1, epochs, i, len(test_loader)), end="\r")
    return acc/n, ssim/n # returns attach success rate
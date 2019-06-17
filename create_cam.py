import os
import torch
from torch.nn import functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from prepare_dataset import load_dataset
from resnet import ResNet18_CAM

device = 'cuda'
dataset_name = 'cifar10'
batch_size = 1
train_data, test_data, in_channels, num_classes = load_dataset(dataset_name)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = ResNet18_CAM()
net.cuda()

checkpoint_path = os.path.join('saved', 'cifar10', 'target_models', 'best_cifar10_cam.pth.tar')
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['state_dict'])
net.eval()
finalconv_name = 'conv'

# hook
feature_blobs = []
def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

params = list(net.parameters())
# get weight only from the last layer(linear)
#weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
weight_softmax = params[-2].cpu().data.numpy()
#print(np.shape(weight_softmax))

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (320, 320)
    bz, nc, h, w = feature_conv.shape # 1, 512, 4, 4
    output_cam = []
    #cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


# test on test set
#image_tensor, _ = next(iter(test_loader))
#image_np = image_tensor.squeeze().numpy()
#image_np = (image_np+1)/2.0
#image_np = np.transpose(image_np, (1, 2, 0))
#image_np = image_np * 255.0
#cv2.imwrite('images/cifar10/cam/test.jpg', image_np)
##image_PIL = transforms.ToPILImage()(image_tensor[0])
##image_PIL.save('images/cifar10/cam/test.jpg')
#
#image_tensor = image_tensor.to(device)
#logit, _ = net(image_tensor)
#h_x = F.softmax(logit, dim=1).data.squeeze()
#probs, idx = h_x.sort(0, True)
#print(idx[0].item(), classes[idx[0]], probs[0].item())
#CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
#img = cv2.imread('images/cifar10/cam/test.jpg')
#height, width, _ = img.shape
#heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
#result = heatmap * 0.3 + img * 0.5
#cv2.imwrite('images/cifar10/cam/CAM.jpg', result)

# test on adversaries
#i=0
#img_fake_orig = cv2.imread('images/cifar10/cam/fake_fgsm/%d_adv.png'%(i)) # [32,32,3]
#height, width, _ = img_fake_orig.shape
#
#img_fake = np.transpose(img_fake_orig, (2, 0, 1))
#img_fake = torch.from_numpy(img_fake).float().to(device) 
#img_fake = torch.unsqueeze(img_fake,0) # [0,255]
#img_fake = img_fake/255.0 # [0,1]
#img_fake = (img_fake*2)-1
#logit, _ = net(img_fake)
#h_x = F.softmax(logit, dim=1).data.squeeze()
#probs, idx = h_x.sort(0, True)
#print(idx[0].item(), classes[idx[0]], probs[0].item())
#CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
#heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET) # [-1,1]
#result = heatmap * 0.3 + img_fake_orig * 0.5
#result = result
#cv2.imwrite('images/cifar10/cam/fake_fgsm/%d_adv_cam.png'%(i), result)



def tile_cam():
    arr = np.zeros((32*4, 32*10, 3), dtype=np.float64)
    for i in range(10):
        path = 'images/cifar10/cam/real_fgsm'
        img_real = '%d.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_real = img_real/255.0
        
        path = 'images/cifar10/cam/real_fgsm_cam'
        img_real_cam = '%d_cam.png'%(i)
        img_path = os.path.join(path, img_real_cam)
        img_real_cam = cv2.imread(img_path, cv2.IMREAD_COLOR)[...,::-1] # convert BGR to RGB
        img_real_cam = img_real_cam/255.0
        
        path = 'images/cifar10/cam/fake_fgsm'
        img_fake = '%d_adv.png'%(i)
        img_path = os.path.join(path, img_fake)
        img_fake = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_fake = img_fake/255.0
        
        path = 'images/cifar10/cam/fake_fgsm_cam'
        img_fake_cam = '%d_adv_cam.png'%(i)
        img_path = os.path.join(path, img_fake_cam)
        img_fake_cam = cv2.imread(img_path, cv2.IMREAD_COLOR)[...,::-1]
        img_fake_cam = img_fake_cam/255.0
        
        print(np.shape(img_real))
        a=0
        b = i
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_real
        a = 1
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_real_cam
        a = 2
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_fake
        a = 3
        arr[a*32: (a+1)*32, b*32: (b+1)*32, :] = img_fake_cam
    plt.figure(figsize=(17,7))
    ax = plt.gca()
    ax.imshow(arr)
    plt.axis('off')
    plt.show() 
    return arr

tile_cam()
##
#img = cv2.imread('images/cifar10/cam/9_cam.png')
#img = img.astype(np.float64)
#img = img/255.0
#plt.figure(figsize=(2,2))
#ax=plt.gca()
#ax.imshow(img)
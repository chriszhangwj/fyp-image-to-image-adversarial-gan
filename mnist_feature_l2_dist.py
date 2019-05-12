import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import target_models
import os
import torch
from tensorflow.examples.tutorials.mnist import input_data
from torch.autograd import Variable
from prepare_dataset import load_dataset
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
#
## training data has 55000 images
#X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
#y_train = mnist.train.labels
#
## test data has 10000 images
#X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
#y_test = mnist.test.labels

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

# load target model
device = 'cuda'
model_name = 'Model_C'
f = getattr(target_models, model_name)(1, 10)
checkpoint_path_f = os.path.join('saved', 'target_models', 'best_%s_mnist_fc64.pth.tar'%(model_name))
checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
f.load_state_dict(checkpoint_f["state_dict"])
f.eval()
f.cuda()

# get features for training set
#train_data, test_data, in_channels, num_classes = load_dataset('mnist')
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
print(np.shape(feature))
#print(np.shape(label_vec))
# print(len(feature))

# different-class nearest-neighbour search
j_idx_dict = np.zeros((10000,1))
for i in range(len(feature)): # i for target image
    print('searching for index %d'%(i))
    i_feature = feature[i,:]
    norm_best = 1e5
    j_best = 0
    
    for j in range(len(feature)):
        if label_vec[j] != label_vec[i]:
            j_feature = feature[j,:]
            # compute L2 norm 
            norm_temp = np.linalg.norm(i_feature-j_feature)
            #print(norm_temp)
            if norm_temp < norm_best:
                j_best = j # update best index so fat 
                norm_best = norm_temp # update smallest norm so far
    j_idx_dict[i] = j_best

j_label_dict = np.zeros((10000,1))
for i in range(len(j_idx_dict)):
    j_label_idx = int(j_idx_dict[i].item()) # get index of current j
    j_label_dict[i] = label_vec[j_label_idx] # get the label of current j
    print(i)

np.savetxt("mnist_j_idx_dict_1e4.csv", j_idx_dict, delimiter=",")
np.savetxt("mnist_j_label_dict_1e4.csv", j_label_dict, delimiter=",")


# prepare data for GAN training
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)





## initialise distance matrix
#dist_mat = np.zeros((10,10))
#for i in range(10):
#    for j in range(10):
#        dist_mat[i,j] = np.linalg.norm(img_mean[i]-img_mean[j]) # compute inter-class L2 distance
#
#digits = ['0','1','2','3','4','5','6','7','8','9',]
#fig, ax = plt.subplots(figsize=(6,6))
#im = ax.imshow(dist_mat)
#
## We want to show all ticks...
#ax.set_xticks(np.arange(len(digits)))
#ax.set_yticks(np.arange(len(digits)))
## ... and label them with the respective list entries
#ax.set_xticklabels(digits)
#ax.set_yticklabels(digits)
#
## Rotate the tick labels and set their alignment.
#plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
#         rotation_mode="anchor")
#
## Loop over data dimensions and create text annotations.
#for i in range(len(digits)):
#    for j in range(len(digits)):
#        text = ax.text(j, i, np.around(dist_mat[i, j],2),
#                       ha="center", va="center", color="w", fontsize=9)
#
##ax.set_title("Harvest of local farmers (in tons/year)")
#fig.tight_layout()
#plt.show()
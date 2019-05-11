import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pytorch_ssim
import torch
from tensorflow.examples.tutorials.mnist import input_data
device = 'cuda'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

# training data has 55000 images
X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
y_train = mnist.train.labels

# test data has 10000 images
X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
y_test = mnist.test.labels

img_0 = X_train[np.where(y_train == 0)[0]]
img_1 = X_train[np.where(y_train == 1)[0]]
img_2 = X_train[np.where(y_train == 2)[0]]
img_3 = X_train[np.where(y_train == 3)[0]]
img_4 = X_train[np.where(y_train == 4)[0]]
img_5 = X_train[np.where(y_train == 5)[0]]
img_6 = X_train[np.where(y_train == 6)[0]]
img_7 = X_train[np.where(y_train == 7)[0]]
img_8 = X_train[np.where(y_train == 8)[0]]
img_9 = X_train[np.where(y_train == 9)[0]]

# compute mean image for each class
img_0_mean = np.mean(img_0,0)
img_1_mean = np.mean(img_1,0)
img_2_mean = np.mean(img_2,0)
img_3_mean = np.mean(img_3,0)
img_4_mean = np.mean(img_4,0)
img_5_mean = np.mean(img_5,0)
img_6_mean = np.mean(img_6,0)
img_7_mean = np.mean(img_7,0)
img_8_mean = np.mean(img_8,0)
img_9_mean = np.mean(img_9,0)
img_mean = np.array([img_0_mean,img_1_mean,img_2_mean,img_3_mean,img_4_mean,img_5_mean,img_6_mean,img_7_mean,img_8_mean,img_9_mean])

# initialise distance matrix
dist_mat = np.zeros((10,10))

# compute interclass distance
for i in range(10):
    for j in range(10):
        img1 = torch.from_numpy(img_mean[i]).float().to(device)
        img2 = torch.from_numpy(img_mean[j]).float().to(device)
        if torch.cuda.is_available():
            img1 = img1.cuda()
            img1 = img1.reshape([-1,1,28,28])
            img2 = img2.cuda()
            img2 = img2.reshape([-1,1,28,28])
            dist_mat[i,j]= pytorch_ssim.ssim(img1,img2)
            
digits = ['0','1','2','3','4','5','6','7','8','9',]
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(dist_mat)

# We want to show all ticks...
ax.set_xticks(np.arange(len(digits)))
ax.set_yticks(np.arange(len(digits)))
# ... and label them with the respective list entries
ax.set_xticklabels(digits)
ax.set_yticklabels(digits)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(digits)):
    for j in range(len(digits)):
        text = ax.text(j, i, np.around(dist_mat[i, j],3),
                       ha="center", va="center", color="w", fontsize=9)

#ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()

# Final selection (find the highest SSIM scores)
# 0 - 5
# 1 - 8
# 2 - 3
# 3 - 5
# 4 - 9
# 5 - 8
# 6 - 2
# 7 - 9
# 8 - 5
# 9 - 4
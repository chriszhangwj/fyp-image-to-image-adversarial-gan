import torch
import pytorch_ssim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import target_models
from generators import Generator_MNIST as Generator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def tsne_plot(data, label, dim=2):
    #data is the feature vector
    #label is the corresponding label
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data)

    tsne = TSNE(n_components=dim, verbose = 1)
    tsne_data = tsne.fit_transform(pca_data)
    
    color_map = label
    
    plt.figure(figsize=(10,10))
    for cl in range(10):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_data[indices,0], tsne_data[indices,1], label=cl)
    plt.legend()
    plt.show()
    return None

#feature_vec = np.genfromtxt("feature_200_mnist.csv", delimiter=',')
#label_vec = np.genfromtxt("mnist_train_label_numpy.csv", delimiter=',')
#print(np.shape(feature_vec))
#print(np.shape(label_vec))
#tsne_plot(feature_vec,label_vec,dim=2)

def stitch_images(images, y_img_count, x_img_count, margin = 2):
    # Define a function that stitches the 28 * 28 numpy arrays
    # together into a collage.
    
    # Example usage:
    #x_sample = x_validation[0].reshape(28, 28)
    #adv_x_sample = adv_x[0].reshape(28, 28)
    #adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2) # need to append images into a list
    #plt.imshow(adv_comparison)
    #plt.show()
    
    # Dimensions of the images
    img_width = images[0].shape[0]
    img_height = images[0].shape[1]    
    width = y_img_count * img_width + (y_img_count - 1) * margin
    height = x_img_count * img_height + (x_img_count - 1) * margin
    stitched_images = np.zeros((width, height))

    # Fill the picture with our saved filters
    for i in range(y_img_count): # 1
        for j in range(x_img_count): # 10
            img = images[i * x_img_count + j]
            #print(len(img.shape))
            #print(img.shape)
            #if len(img.shape) == 2:
            #    img = np.dstack([img] * 3)
            stitched_images[(img_width+margin)*i:(img_width+margin)*i+img_width, (img_height+margin)*j:(img_height+margin)*j+img_height] = img
    return stitched_images

# Take an image in the form of a numpy array and return it with a coloured border
def add_border(digit_img, border_color = 'black', margin = 1):
    digit_shape = digit_img.shape
    print(digit_shape) # 28*28
    base = np.zeros((digit_shape[0] + 2 * margin, digit_shape[1] + 2 * margin, 3))
    rgb_digit = np.dstack([digit_img] * 3)
    
    if border_color == 'red':
        base[:,:,0] = 1
    elif border_color == 'green':
        base[:,:,1] = 1
    elif border_color == 'yellow':
        base[:,:,0] = 1
        base[:,:,1] = 1
    
    border_digit = base
    print(border_digit.shape)
    border_digit[margin:(digit_shape[0] + 1), margin:(digit_shape[1] + 1), :] = rgb_digit
    
    return base

def plot_grid_targeted(model_name, border=True): # plot 10 by 10 grid for targeted attack
    thres = 0.3
    rows = []
    for i in range(10): # digit to be attacked
        #load image to be attacked
        results = []
        digit = i
        img_path = 'images/%d.jpg'%(digit)
        orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = orig.copy().astype(np.float32)
        img = img[None, None, :, :]/255.0                        
        x = torch.from_numpy(img) 
            
        for j in range(10): # target digit
            if i == j:
                img_x = x.data.squeeze().numpy()
                results.append(img_x)
            else:
                # load the generator for current target class
                target = j 
                G = Generator()
                checkpoint_name_G = '%s_target_%d.pth.tar'%(model_name, target)
                checkpoint_path_G = os.path.join('saved', 'generators', 'bound_%.1f'%(thres), checkpoint_name_G)
                #checkpoint_path_G = os.path.join('saved','Model_C_untargeted.pth.tar')
                checkpoint_G = torch.load(checkpoint_path_G, map_location='cpu')
                G.load_state_dict(checkpoint_G['state_dict'])
                G.eval()
                # compute perturbed image for current target  class
                pert = G(x).data.clamp(min=-thres, max=thres)
                x_adv = x + pert 
                x_adv = x_adv.clamp(min=0, max=1)
                img_adv = x_adv.data.squeeze().numpy() 
                results.append(img_adv)
        results_img = stitch_images(results, 1, 10, margin = 0)
        rows.append(results_img)

    # Plot the resulting grid
    final_img = stitch_images(rows, 10, 1, margin = 0)
    plt.figure(figsize=(8,8))
    plt.imshow(final_img, cmap = 'gray')
    plt.xticks([])
    plt.xlabel("Target Digits")
    plt.yticks([])
    plt.ylabel("Input Digits")
    plt.show()
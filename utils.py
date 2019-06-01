import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
from generators import Generator_MNIST as Generator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from noise import pnoise2
torch.manual_seed(0)

def tsne_plot(data, label, dim=2):
    #data is the feature vector
    #label is the corresponding label
    pca = PCA(n_components=10)
    pca_data = pca.fit_transform(data)

    tsne = TSNE(n_components=dim, perplexity=50, verbose = 1)
    tsne_data = tsne.fit_transform(pca_data)
    
    color_map = label
    
    plt.figure(figsize=(7,7))
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
    
def tile_results():
    images = os.listdir('results/')
    images.sort()

    arr = np.zeros((28*10, 28*10), dtype=np.uint8)

    for i, image in enumerate(images):
        img_path = os.path.join('results', image)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        a = int(image.split('_')[0])
        b = int(image.split('_')[1].split('.')[0])

        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img

    import pdb; pdb.set_trace()
    
def tile_evolution():
    arr = np.zeros((28*10, 28*11), dtype=np.uint8)
    for i in range(10):
        path = 'images/train_evolution/%d/'%(i)
        images = os.listdir(path)
        images.sort()
            
        for j in range(11): # loop epochs
            img_fake = '%d_epoch_%d.png'%(i,j)
            img_path = os.path.join(path, img_fake)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            a = int(img_fake.split('_')[0]) # split string and obtain key word
            b = int(img_fake.split('_')[2].split('.')[0])
            arr[a*28: (a+1)*28, b*28: (b+1)*28] = img
            
            
    plt.figure(figsize=(7,7))
    plt.imshow(arr, cmap = 'gray')
    plt.axis('off')
    plt.show()    
    
def plot_pert():
    arr = np.zeros((28*3, 28*10), dtype=np.int16)
    for i in range(10):
        path = 'images/train_evolution/%d'%(i)
        images = os.listdir(path)
        images.sort()
        
        img_real = '%d_epoch_0.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_adv = '%d_epoch_10.png'%(i)
        img_path = os.path.join(path, img_adv)
        img_fake = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_real = img_real.astype(np.int16)
        img_fake = img_fake.astype(np.int16)
        
        a=0
        b = i
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_real
        img_pert = abs(img_fake - img_real)
        a = 1
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_pert
        a = 2
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_fake
            
    plt.figure(figsize=(20,3))
    ax = plt.gca()
    im = ax.imshow(arr)
    plt.axis('off')
    plt.colorbar(im)
    plt.show()    
    
    # show un-normalised perturbation map  
    arr = np.zeros((28*1, 28*10), dtype=np.int16)
    for i in range(10):
        path = 'images/train_evolution/%d'%(i)
        img_real = '%d_epoch_0.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_fake = '%d_epoch_10.png'%(i)
        img_path = os.path.join(path, img_fake)
        img_fake = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_real = img_real.astype(np.int16)
        img_fake = img_fake.astype(np.int16)
        img_pert = img_fake - img_real # [-255,255]        
        a = 0
        b = i
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_pert
    
    plt.figure(figsize=(20,1))
    ax = plt.gca()
    im = ax.imshow(arr,cmap='coolwarm')
    plt.axis('off')
    plt.colorbar(im)
    plt.show()  
    
def plot_pert_advgan():
    arr = np.zeros((28*3, 28*10), dtype=np.int16)
    for i in range(10):
        path = 'images/advgan/'
        images = os.listdir(path)
        images.sort()
        
        img_real = '%d.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img_fake = '%d_adv.png'%(i)
        img_path = os.path.join(path, img_fake)
        img_fake = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_real = img_real.astype(np.int16)
        img_fake = img_fake.astype(np.int16)
        
        a=0
        b = i
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_real
        img_pert = abs(img_fake - img_real)
        a=1
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_pert
        a = 2
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_fake
            
    plt.figure(figsize=(20,3))
    ax = plt.gca()
    im = ax.imshow(arr)
    plt.axis('off')
    plt.colorbar(im)
    plt.show()   
    
  
    # show un-normalised perturbation map  
    arr = np.zeros((28*1, 28*10), dtype=np.int16)
    for i in range(10):
        img_real = '%d.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img_fake = '%d_adv.png'%(i)
        img_path = os.path.join(path, img_fake)
        img_fake = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_real = img_real.astype(np.int16)
        img_fake = img_fake.astype(np.int16)
        img_pert = img_fake - img_real # [-255,255]    
        a = 0
        b = i
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_pert
    
    plt.figure(figsize=(20,1))
    ax = plt.gca()
    im = ax.imshow(arr,cmap='coolwarm')
    plt.axis('off')
    plt.colorbar(im)
    im.set_clim(-255,255)
    plt.show()  
    
def plot_pert_cw():
    arr = np.zeros((28*3, 28*10), dtype=np.int16)
    for i in range(10):
        path = 'images/cw/'
        images = os.listdir(path)
        images.sort()
        
        img_real = '%d.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img_fake = '%d_adv.png'%(i)
        img_path = os.path.join(path, img_fake)
        img_fake = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_real = img_real.astype(np.int16)
        img_fake = img_fake.astype(np.int16)
        
        a=0
        b = i
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_real
        img_pert = abs(img_fake - img_real)
        a=1
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_pert
        a = 2
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_fake
            
    plt.figure(figsize=(20,3))
    ax = plt.gca()
    im = ax.imshow(arr)
    plt.axis('off')
    plt.colorbar(im)
    plt.show()   
    
  
    # show un-normalised perturbation map  
    arr = np.zeros((28*1, 28*10), dtype=np.int16)
    for i in range(10):
        img_real = '%d.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img_fake = '%d_adv.png'%(i)
        img_path = os.path.join(path, img_fake)
        img_fake = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_real = img_real.astype(np.int16)
        img_fake = img_fake.astype(np.int16)
        img_pert = img_fake - img_real # [-255,255]    
        a = 0
        b = i
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_pert
    
    plt.figure(figsize=(20,1))
    ax = plt.gca()
    im = ax.imshow(arr,cmap='coolwarm')
    plt.axis('off')
    plt.colorbar(im)
    im.set_clim(-255,255)
    plt.show()  
    
def plot_pert_deepfool():
    arr = np.zeros((28*3, 28*10), dtype=np.int16)
    for i in range(10):
        path = 'images/deepfool/'
        images = os.listdir(path)
        img_real = '%d.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_fake = '%d_adv.png'%(i)
        img_path = os.path.join(path, img_fake)
        img_fake = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_real = img_real.astype(np.int16)
        img_fake = img_fake.astype(np.int16)
        a=0
        b = i
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_real
        img_pert = abs(img_fake - img_real)
        a=1
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_pert
        a=2
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_fake     
    plt.figure(figsize=(20,3))
    ax = plt.gca()
    im = ax.imshow(arr)
    plt.axis('off')
    plt.colorbar(im)
    plt.show()   
    # show un-normalised perturbation map  
    arr = np.zeros((28*1, 28*10), dtype=np.int16)
    for i in range(10):
        img_real = '%d.png'%(i)
        img_path = os.path.join(path, img_real)
        img_real = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        img_fake = '%d_adv.png'%(i)
        img_path = os.path.join(path, img_fake)
        img_fake = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_real = img_real.astype(np.int16)
        img_fake = img_fake.astype(np.int16)
        img_pert = img_fake - img_real # [-255,255]    
        a = 0
        b = i
        arr[a*28: (a+1)*28, b*28: (b+1)*28] = img_pert
    
    plt.figure(figsize=(20,1))
    ax = plt.gca()
    im = ax.imshow(arr,cmap='coolwarm')
    plt.axis('off')
    plt.colorbar(im)
    im.set_clim(-255,255)
    plt.show()  

def perlin(size, period, octave, freq_sine, lacunarity = 2): # Perlin noise with sine color map
    
    # Perlin noise
    noise = np.empty((size, size), dtype = np.float32)
    for x in range(size):
        for y in range(size):
            noise[x][y] = pnoise2(x / period, y / period, octaves = octave, lacunarity = lacunarity)
            
    # Sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    return normalize(noise)

def normalize(vec): # Normalize vector
    vmax = np.amax(vec)
    vmin  = np.amin(vec)
    return (vec - vmin) / (vmax - vmin)

def colorize(img, color = [1, 1, 1]): # colorize perlin noise for coloured images
    if img.ndim == 2: # expand to include color channels
        img = np.expand_dims(img, 2)
    return (img - 0.5) * color + 0.5 # output pixel range [0, 1]

def toZeroThreshold(x, t=0.5):
    zeros = torch.cuda.FloatTensor(x.shape).fill_(0.0)
    ones = torch.cuda.FloatTensor(x.shape).fill_(1.0)
    return torch.where(x > t, ones, zeros)

def cw_l2(model, images, labels, c=1e-6, kappa=0, max_iter=400, learning_rate=0.01) :
    device = 'cuda'
    images = images.to(device)     
    labels = labels.to(device)
    # Define f-function
    def f(x) :
        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())        
        # If untargeted, optimize for making the other class most likely 
        return torch.clamp(j-i, min=-kappa)    
    w = torch.zeros_like(images, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    prev = 1e10    
    for step in range(max_iter) :
        a = 1/2*(nn.Tanh()(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost    
    attack_images = 1/2*(nn.Tanh()(w) + 1)
    return attack_images

class Attacker:
    def __init__(self, clip_max=1, clip_min=0):
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        pass

class DeepFool(Attacker):
    def __init__(self, max_iter=50, clip_max=1, clip_min=0):
        super(DeepFool, self).__init__(clip_max, clip_min)
        self.max_iter = max_iter

    def generate(self, model, x, y, device):
        nx = Variable(x.to(device)) # image
        nx.requires_grad_()
        eta = torch.zeros(nx.shape) # perturbation
        eta = Variable(eta.to(device))      
        out = model(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()
        i_iter = 0
        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None
            for i in range(n_class):
                if i == py:
                    continue
                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                fi = fi.cpu()
                wi = wi.cpu()
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())
                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi
                
            ri = Variable(ri.to(device))
            eta += ri.clone()
            nx.grad.data.zero_()
            temp = torch.clamp(nx+eta,0,1)
            out = model(temp)
            py = out.max(1)[1].item()
            i_iter += 1
            if i_iter+1 == self.max_iter:
                print('failed to converge')      
        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        return x_adv.detach()
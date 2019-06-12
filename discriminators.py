import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
import functools


class Discriminator_MNIST(nn.Module):
	def __init__(self):
		super(Discriminator_MNIST, self).__init__()
		self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
		#self.in1 = nn.InstanceNorm2d(8)
		# "We do not use instanceNorm for the first C8 layer."
		self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
		self.in2 = nn.InstanceNorm2d(16)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		self.in3 = nn.InstanceNorm2d(32)
		self.fc = nn.Linear(3*3*32, 1)
        
	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
		x = F.leaky_relu(self.in2(self.conv2(x)), negative_slope=0.2)
		x = F.leaky_relu(self.in3(self.conv3(x)), negative_slope=0.2)
		x = x.view(x.size(0), -1)
		x = self.fc(x) # output is one digit representing real or fake
		return x
    
class Discriminator_CIFAR10(nn.Module):
	def __init__(self):
		super(Discriminator_CIFAR10, self).__init__()
		self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)
		#self.in1 = nn.InstanceNorm2d(8)
		# "We do not use instanceNorm for the first C8 layer."
		self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
		self.in2 = nn.InstanceNorm2d(16)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		self.in3 = nn.InstanceNorm2d(32)
		self.fc = nn.Linear(4*4*32, 1)
        
	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
		x = F.leaky_relu(self.in2(self.conv2(x)), negative_slope=0.2)
		x = F.leaky_relu(self.in3(self.conv3(x)), negative_slope=0.2)
		x = x.view(x.size(0), -1)
		x = self.fc(x) # output is one digit representing real or fake
		return x
    
class Discriminator_ACGAN(nn.Module): # use LeNet-5 used as auxiliary classifier
	def __init__(self):
		super(Discriminator_ACGAN, self).__init__()
		self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
		self.in2 = nn.InstanceNorm2d(16)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		self.in3 = nn.InstanceNorm2d(32)
		self.fc1 = nn.Linear(3*3*32, 1) # convert fc layer to 1-dim output
		self.fc2 = nn.Linear(3*3*32, 10)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()
        
	def forward(self, x):
		#x = self.noise(x)
		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
		x = F.leaky_relu(self.in2(self.conv2(x)), negative_slope=0.2)
		x = F.leaky_relu(self.in3(self.conv3(x)), negative_slope=0.2)
		x = x.view(x.size(0), -1)
		x1 = self.fc1(x) # real vs fake
		x1 = self.sigmoid(x1)
		x2 = self.fc2(x) # class distribution (auxiliary)
		x2 = self.softmax(x2)
		return x1, x2
    
class Discriminator_ACGAN2(nn.Module): # use LeNet-5 used as auxiliary classifier
	def __init__(self):
		super(Discriminator_ACGAN2, self).__init__()
		self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
		self.fc1 = nn.Linear(3*3*16, 1) # convert fc layer to 1-dim output
		self.fc2 = nn.Linear(3*3*16, 10)
		self.dropout = nn.Dropout(0.5)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()
		self.in2 = nn.InstanceNorm2d(16)
		self.in3 = nn.InstanceNorm2d(32)
		self.bn1 = nn.BatchNorm2d(4)
		self.bn2 = nn.BatchNorm2d(8)
		self.bn3 = nn.BatchNorm2d(16)
        
	def forward(self, x):
		#x = self.noise(x)
#		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
#		x = self.dropout(x)
#		x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
#		x = self.dropout(x)
#		x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
#		x = self.dropout(x)
        
		x = self.bn1(F.leaky_relu(self.conv1(x), negative_slope=0.2))
		x = self.dropout(x) 
		x = self.bn2(F.leaky_relu(self.conv2(x), negative_slope=0.2))
		x = self.dropout(x)
		x = self.bn3(F.leaky_relu(self.conv3(x), negative_slope=0.2))
		x = self.dropout(x)
		x = x.view(x.size(0), -1) # flatten layer
		x1 = self.fc1(x) # real vs fake
		x2 = self.fc2(x) # class distribution (auxiliary)
		x2 = self.softmax(x2)
		return x1, x2
    
class Discriminator_ACGAN2_CIFAR10(nn.Module): # use LeNet-5 used as auxiliary classifier
	def __init__(self):
		super(Discriminator_ACGAN2_CIFAR10, self).__init__()
		self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		self.fc1 = nn.Linear(4*4*32, 1) # convert fc layer to 1-dim output
		self.fc2 = nn.Linear(4*4*32, 10)
		self.dropout = nn.Dropout(0.5)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()
		self.in2 = nn.InstanceNorm2d(16)
		self.in3 = nn.InstanceNorm2d(32)
		self.bn1 = nn.BatchNorm2d(8)
		self.bn2 = nn.BatchNorm2d(16)
		self.bn3 = nn.BatchNorm2d(32)
        
	def forward(self, x):
		#x = self.noise(x)
#		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
#		x = self.dropout(x)
#		x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
#		x = self.dropout(x)
#		x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
#		x = self.dropout(x)
        
		x = self.bn1(F.leaky_relu(self.conv1(x), negative_slope=0.2))
		x = self.dropout(x) 
		x = self.bn2(F.leaky_relu(self.conv2(x), negative_slope=0.2))
		x = self.dropout(x)
		x = self.bn3(F.leaky_relu(self.conv3(x), negative_slope=0.2))
		x = self.dropout(x)
		x = x.view(x.size(0), -1) # flatten layer
		x1 = self.fc1(x) # real vs fake
		x2 = self.fc2(x) # class distribution (auxiliary)
		x2 = self.softmax(x2)
		return x1, x2
    
def PatchGAN(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02):
    """Create a discriminator
    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
    Returns a discriminator
    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.
        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain)  

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 3
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer 

def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net 
    
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) # apply the initialization function <init_func>
    
class Identity(nn.Module):
    def forward(self, x):
        return x    

#class ResBlock(nn.Module):
#    
#    def __init__(self, in_features, norm=False):
#        super(ResBlock, self).__init__()
#
#        block = [  nn.ReflectionPad2d(1),
#                nn.Conv2d(in_features, in_features, 3),
#                # nn.InstanceNorm2d(in_features),
#                nn.ReLU(inplace=True),
#                nn.ReflectionPad2d(1),
#                nn.Conv2d(in_features, in_features, 3),
#                # nn.InstanceNorm2d(in_features)
#                ]
#
#        if norm:
#            block.insert(2,  nn.InstanceNorm2d(in_features))
#            block.insert(6,  nn.InstanceNorm2d(in_features))
#
#
#        self.model = nn.Sequential(*block)
#
#    def forward(self, x):
#        return x + self.model(x)
#
#
#if __name__ == '__main__':
#
#	from tensorboardX import SummaryWriter
#	from torch.autograd import Variable
#
#	X = Variable(torch.rand(13, 1, 28, 28))
#
#	model = Discriminator_MNIST()
#	model(X)
#
#	with SummaryWriter(log_dir="visualization/Discriminator_MNIST", comment='Discriminator_MNIST') as w:
#		w.add_graph(model, (X, ), verbose=True)
#
#class GaussianNoise(nn.Module):
#	"""Gaussian noise regularizer.
#
#	Args:
#		sigma (float, optional): relative standard deviation used to generate the
#		noise. Relative means that it will be multiplied by the magnitude of
#	 	the value your are adding the noise to. This means that sigma can be
#		same regardless of the scale of the vector.
#		is_relative_detach (bool, optional): whether to detach the variable before
#	 	computing the scale of the noise. If `False` then the scale of the noise
#	 	 won't be seen as a constant but something to optimize: this will bias the
#		network to generate vectors with smaller values.
#	"""
#	def __init__(self, sigma=0.1, is_relative_detach=True):
#		super().__init__()
#		self.sigma = sigma
#		self.is_relative_detach = is_relative_detach
#		self.noise = torch.tensor(0).to(device)
#
#	def forward(self, x):
#		if self.training and self.sigma != 0:
#			scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
#			sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
#			x = x + sampled_noise
#		return x 

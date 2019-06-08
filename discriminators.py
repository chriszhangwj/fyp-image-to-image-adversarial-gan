import torch.nn as nn
import torch
import torch.nn.functional as F


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
		self.conv1 = nn.Conv2d(3, 4, kernel_size=4, stride=2, padding=1)
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
    
    
class Attn(nn.Module):
    def __init__(self, input_nc=1):
        super(Attn, self).__init__()
        model =  [  nn.Conv2d(1, 8, 4, stride=1, padding=2),
                    nn.InstanceNorm2d(8),
                    nn.ReLU(inplace=True) ]

        model += [  nn.Conv2d(8, 16, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(16),
                    nn.ReLU(inplace=True) ]

        model += [ResBlock(16, norm=True)]

        model += [nn.UpsamplingNearest2d(scale_factor=2)]

        model += [  nn.Conv2d(16, 16, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(16),
                    nn.ReLU(inplace=True) ]

        model += [  nn.Conv2d(16, 8, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True) ]

        model += [  nn.Conv2d(8, 1, 3, stride=1, padding=1),
                    nn.Sigmoid() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    
    def __init__(self, in_features, norm=False):
        super(ResBlock, self).__init__()

        block = [  nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                # nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                # nn.InstanceNorm2d(in_features)
                ]

        if norm:
            block.insert(2,  nn.InstanceNorm2d(in_features))
            block.insert(6,  nn.InstanceNorm2d(in_features))


        self.model = nn.Sequential(*block)

    def forward(self, x):
        return x + self.model(x)


if __name__ == '__main__':

	from tensorboardX import SummaryWriter
	from torch.autograd import Variable

	X = Variable(torch.rand(13, 1, 28, 28))

	model = Discriminator_MNIST()
	model(X)

	with SummaryWriter(log_dir="visualization/Discriminator_MNIST", comment='Discriminator_MNIST') as w:
		w.add_graph(model, (X, ), verbose=True)

class GaussianNoise(nn.Module):
	"""Gaussian noise regularizer.

	Args:
		sigma (float, optional): relative standard deviation used to generate the
		noise. Relative means that it will be multiplied by the magnitude of
	 	the value your are adding the noise to. This means that sigma can be
		same regardless of the scale of the vector.
		is_relative_detach (bool, optional): whether to detach the variable before
	 	computing the scale of the noise. If `False` then the scale of the noise
	 	 won't be seen as a constant but something to optimize: this will bias the
		network to generate vectors with smaller values.
	"""
	def __init__(self, sigma=0.1, is_relative_detach=True):
		super().__init__()
		self.sigma = sigma
		self.is_relative_detach = is_relative_detach
		self.noise = torch.tensor(0).to(device)

	def forward(self, x):
		if self.training and self.sigma != 0:
			scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
			sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
			x = x + sampled_noise
		return x 

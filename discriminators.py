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
		self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
		self.fc1 = nn.Linear(3*3*32, 1) # convert fc layer to 1-dim output
		self.fc2 = nn.Linear(3*3*32, 10)
		self.dropout = nn.Dropout(0.5)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()
        
	def forward(self, x):
		#x = self.noise(x)
		x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
		#x = self.dropout(x)
		x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
		#x = self.dropout(x)
		x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
		#x = self.dropout(x)
		x = x.view(x.size(0), -1) # flatten layer
		x1 = self.fc1(x) # real vs fake
		#x1 = self.sigmoid(x1)
		x2 = self.fc2(x) # class distribution (auxiliary)
		x2 = self.softmax(x2)
		return x1, x2


if __name__ == '__main__':

	from tensorboardX import SummaryWriter
	from torch.autograd import Variable
	from torchvision import models

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

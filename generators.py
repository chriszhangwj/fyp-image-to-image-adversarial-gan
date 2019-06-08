import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.conv2d(x)
        return x

class Generator_MNIST(nn.Module):
    def __init__(self):
        super(Generator_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(32)

        self.resblock1 = ResidualBlock(32)
        self.resblock2 = ResidualBlock(32)
        self.resblock3 = ResidualBlock(32)
        self.resblock4 = ResidualBlock(32)

        self.up1 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(16)
        self.up2 = UpsampleConvLayer(16, 8, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(8)

        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.in6 = nn.InstanceNorm2d(1) # originally self.in6 = nn.InstanceNorm2d(8) but we think it's a mistake


    def forward(self, x):

        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.in4(self.up1(x))
        x = self.in5(self.up2(x))

        x = self.conv4(x) # remove relu for better performance and when input is [-1 1]
        return x
    
class Generator_MNIST2(nn.Module):
    def __init__(self):
        super(Generator_MNIST2, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(64)

        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)
        self.resblock4 = ResidualBlock(64)

        self.up1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(32)
        self.up2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(16)

        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        #self.in6 = nn.InstanceNorm2d(1) # originally self.in6 = nn.InstanceNorm2d(8) but we think it's a mistake


    def forward(self, x):
        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = F.relu(self.in4(self.up1(x)))
        x = F.relu(self.in5(self.up2(x)))
        x = self.conv4(x) # remove relu for better performance and when input is [-1 1]
        return x
    
class Generator_CIFAR10(nn.Module):
    def __init__(self):
        super(Generator_CIFAR10, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(32)

        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)
        self.resblock4 = ResidualBlock(64)

        self.up1 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(32)
        self.up2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(16)

        self.conv4 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.in6 = nn.InstanceNorm2d(3) # originally self.in6 = nn.InstanceNorm2d(8) but we think it's a mistake


    def forward(self, x):

        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.in4(self.up1(x))
        x = self.in5(self.up2(x))

        x = self.conv4(x) # remove relu for better performance and when input is [-1 1]
        return x
    
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=8, output_nc=8, num_downs=3, ngf=4, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 1):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

if __name__ == '__main__':

    from tensorboardX import SummaryWriter
    from torch.autograd import Variable
    from torchvision import models

    X = Variable(torch.rand(13, 1, 28, 28))

    model = Generator_MNIST()

    with SummaryWriter(log_dir="visualization/Generator_MNIST", comment='Generator_MNIST') as w:
        w.add_graph(model, (X, ))

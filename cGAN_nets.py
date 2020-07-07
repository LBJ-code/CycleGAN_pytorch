import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import num_flat_features


class ResnetGenerator(nn.Module):
    """Resnetベースのジェネレータ

    Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
    """

    def __init__(self, input_nc, output_fc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        super(ResnetGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        multi = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * multi, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_fc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)


    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    """ResnetGenerator用のリズネットブロック"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        """Construct a convolutional block.

            Parameters:
                dim (int)           -- the number of channels in the conv layer.
                padding_type (str)  -- the name of padding layer: reflect | replicate | zero
                norm_layer          -- normalization layer
                use_dropout (bool)  -- if use dropout layers.
                use_bias (bool)     -- if the conv layer uses bias or not

            Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

'''
from matplotlib import pyplot as plt
resnet_generator = ResnetGenerator(3, 3)
input_img = torch.randn(1, 3, 352, 480)
fake_images = resnet_generator(input_img)

img_transformed = fake_images[0].detach().numpy().reshape(352, 480, -1)
plt.imshow(img_transformed)
plt.show()
'''

class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

            Parameters:
                input_nc (int)  -- the number of channels in input images
                ndf (int)       -- the number of filters in the last conv layer
                n_layers (int)  -- the number of conv layers in the discriminator
                norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_multi = 1
        nf_multi_prev = 1
        for n in range(1, n_layers):
            nf_multi_prev = nf_multi
            nf_multi = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_multi),
                nn.LeakyReLU(0.2, True)
            ]

        nf_multi_prev = nf_multi
        nf_multi = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_multi, 1, kernel_size=kw, stride=1, padding=padw)]
        self.conv_layers = nn.Sequential(*sequence)


    def forward(self, input):
        conv_out = self.conv_layers(input)
        flat_conv_out = conv_out.view(-1, num_flat_features(conv_out))
        out_fc_layer = nn.Sigmoid()(nn.Linear(num_flat_features(conv_out), 1)(flat_conv_out))

        return out_fc_layer

'''
discriminator = NLayerDiscriminator(3, n_layers=5)
out = discriminator(fake_images)
print(out.shape)
'''

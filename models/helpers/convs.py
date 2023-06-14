# 
# This file is part of the nn_3D-anomaly-detection distribution (https://github.com/snipdome/nn_3D-anomaly-detection).
# Copyright (c) 2022-2023 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
import torch.distributions as dist


class ConvBlock(pl.LightningModule):
    def __init__(self, in_size, out_size, padding=1, batch_norm=False, activation='prelu', dropout=None, is_single_conv=False, flat_latent=True, z_dim=None, kernel_size=3, bias=True, stride=1):
        super(ConvBlock, self).__init__()
        if len(activation)==2:
            if is_single_conv:
                raise Exception('You cannot have two activation functions with a single conv')
            else:
                activation1, activation2 = activation[0].casefold(), activation[1].casefold()
        else:
            activation1 = activation2 = activation.casefold() 

        block = []
        block.append(nn.Conv3d(in_size, out_size, kernel_size=kernel_size, padding=padding, padding_mode='replicate', stride=stride, bias=bias))
        if   activation1 == 'relu':
            nn.init.kaiming_normal_(block[-1].weight)
            block.append(nn.ReLU())
        if   activation1 == 'prelu':
            nn.init.kaiming_normal_(block[-1].weight)
            block.append(nn.PReLU())
        elif activation1 == 'leakyrelu':
            nn.init.kaiming_normal_(block[-1].weight)
            block.append(nn.LeakyReLU(0.01))
        elif activation1 == 'sigmoid':
            nn.init.xavier_uniform_(block[-1].weight) # works well for sigmoid
            block.append(nn.Sigmoid())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        if not is_single_conv:
            block.append(nn.Conv3d(out_size, out_size, kernel_size=kernel_size, padding=padding, padding_mode='replicate', stride=stride,bias=bias))
            if   activation2 == 'relu':
                nn.init.kaiming_normal_(block[-1].weight)
                block.append(nn.ReLU())
            if   activation2 == 'prelu':
                nn.init.kaiming_normal_(block[-1].weight)
                block.append(nn.PReLU())
            elif activation2 == 'leakyrelu':
                nn.init.kaiming_normal_(block[-1].weight)
                block.append(nn.LeakyReLU(0.01))
            elif activation2 == 'sigmoid':
                nn.init.xavier_uniform_(block[-1].weight) # works well for sigmoid
                block.append(nn.Sigmoid())
            if batch_norm:
                block.append(nn.BatchNorm3d(out_size))
            if (dropout is not None) and (dropout):
                block.append(nn.Dropout3d(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class ConvDownBlock(pl.LightningModule):
    def __init__(self, in_size, out_size, down_mode='avg', activation='LeakyRelu', batch_norm=False, padding=1, dropout=None, scale_factor=2, is_single_conv=False, flat_latent=True, kernel_size=3):
        super(ConvDownBlock, self).__init__()
        propagated_vars = {'in_size':in_size, 'out_size':out_size, 'activation':activation, 'batch_norm':batch_norm, 'padding':padding, 'dropout':dropout, 'is_single_conv':is_single_conv, 'kernel_size':kernel_size}
        self.conv_block = ConvBlock(**propagated_vars)
        if scale_factor != 1:
            if down_mode in ('average','avg'):
                self.down_layer = nn.AvgPool3d(scale_factor)
            elif 'max':
                self.down_layer = nn.MaxPool3d(scale_factor)
            else:
                raise Exception('The requested down_mode has not been implemented')
        else:
            self.down_layer = nn.Sequential()

    def forward(self, x):
        out = self.conv_block(x)
        x = self.down_layer(out)
        return x

class ConvUpBlock(pl.LightningModule):
    def __init__(self, in_size, out_size, up_mode='upconv', activation='LeakyRelu', batch_norm=False, padding=1, dropout=None, scale_factor=2, is_single_conv=False, bridge_size=None, flat_latent=True, kernel_size=3, bias=False, stride=1):
        '''
        @param bridge_size: If the bridge has the same number of channel of the desired number of channel of the output, bridge_size is automatically inferred and would be
            the same number of channels of the input (which would be double the size of the output number of channels). 
        '''
        super(ConvUpBlock, self).__init__()
        propagated_vars = {'activation':activation, 'batch_norm':batch_norm, 'padding':padding, 'dropout':dropout, 'is_single_conv':is_single_conv, 'kernel_size':kernel_size, 'bias':bias, 'stride':stride}
        
        if scale_factor != 1:
            if up_mode == 'upconv':
                self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=kernel_size, stride=scale_factor)
            elif up_mode == 'upsample':
                self.up = nn.Sequential(nn.Upsample(mode='trilinear', scale_factor=scale_factor, align_corners=False),
                                        nn.Conv3d(in_size, out_size, kernel_size=1))
            else:
                raise Exception('The requested up_mode has not been implemented')
        else:
            self.up = nn.Sequential()

        if bridge_size is None:
            bridge_size=out_size

        self.conv_block = ConvBlock(bridge_size+out_size, out_size, **propagated_vars)

    def center_crop(self, layer, target_size):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_z = (layer_depth - target_size[0]) // 2
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        return layer[:, :, diff_z:(diff_z + target_size[0]), diff_y:(diff_y + target_size[1]), diff_x:(diff_x + target_size[2])]

    def forward(self, x, bridge=None):
        out = self.up(x)
        if bridge is not None:
            out = torch.cat([out, bridge], 1)
        x = self.conv_block(out)

        return x

class ConvUp(pl.LightningModule):
    def __init__(self, in_size, out_size, up_mode='upconv', activation='LeakyRelu', batch_norm=False, padding=0, dropout=None, scale_factor=2, kernel_size=3, bias=False, stride=1):
        '''
        @param bridge_size: If the bridge has the same number of channel of the desired number of channel of the output, bridge_size is automatically inferred and would be
            the same number of channels of the input (which would be double the size of the output number of channels). 
        '''
        super(ConvUp, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        else:
            raise Exception('The requested up_mode has not been implemented')
        

    def forward(self, x, bridge=None):
        out = self.up(x)
        return out

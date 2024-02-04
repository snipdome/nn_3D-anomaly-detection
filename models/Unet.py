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

import os, pathlib, gc, time, numpy as np, nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
import torchio as tio
import pytorch_lightning as pl
from PIL import Image

from matplotlib import pyplot as plt
import wandb
from PIL import Image
from utils.stats import *
import utils.wandb as wutils
from utils.extern import *
from nn.models.helpers import common_operations as common
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

class Unet(pl.LightningModule):
     
    def __init__(self, name, checkpoint_path, log_path=None, n_channels=1, n_classes=1, depth=3, wf=6, channel_depths=None, padding=True, 
                 batch_norm=False, up_mode='upsample', insta_norm=None, scale_factor=2, stride=1, input_size=256, activation='LeakyReLu',loss={'name': 'cross_entropy'}, evaluate_metrics={}, dropout=False, last_activation='', optimizer_parameters=None, optimizer = 'RMSprop',**kwargs):
        super(Unet, self).__init__()
        self.save_hyperparameters()
        assert up_mode in ('upconv', 'upsample')
        self.name = name
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.checkpoint_path = checkpoint_path
        self.padding = padding
        self.log_path = log_path
        self.depth = depth
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_param = optimizer_parameters
        self.loss = loss
        self.eval_metrics = evaluate_metrics
        self.scale_factor = scale_factor
        
        if not isinstance(input_size, list):
            self.dims = 3
        elif len(input_size) == 2:
            self.dims = 2
            assert (input_size[0]==input_size[1]), 'Wrong dimensions in input'
            input_size = input_size[0]
        elif len(input_size) == 3:
            assert (input_size[0]==input_size[1] and input_size[0]==input_size[2]), 'Wrong dimensions in input'
            input_size = input_size[0]
        
        if channel_depths is None:
            channel_depths = [2**(i+wf) for i in range(depth)]
            
        dropouts = dropout if isinstance(dropout, list) else [dropout for i in range(len(channel_depths))]
        batch_norms = batch_norm if isinstance(batch_norm, list) else [batch_norm for i in range(len(channel_depths))]
        strides = stride if isinstance(stride, list) else [stride for i in range(len(channel_depths))]
        
        self.down_path = nn.ModuleList()
        for i,end_channels in enumerate(channel_depths):
            if i != 0:
                self.down_path.append(UNetConvBlock(channel_depths[i-1], end_channels,
                                                padding=padding, batch_norm=batch_norms[i], activation=activation, 
                                                dropout=dropouts[i], stride=strides[i]))
            else:
                self.down_path.append(UNetConvBlock(n_channels, end_channels,
                                                padding=padding, batch_norm=batch_norms[i], activation=activation, 
                                                dropout=dropouts[i], stride=strides[i]))
        
        self.up_path = nn.ModuleList()
        for i,end_channels in enumerate(reversed(channel_depths[:-1])):
            self.up_path.append(UNetUpBlock(channel_depths[-1-i], end_channels, up_mode=up_mode, 
                                            padding=padding, batch_norm=batch_norms[i], activation=activation, 
                                            dropout=dropouts[i],scale_factor=scale_factor*strides[i]))

        self.last = nn.Conv3d(channel_depths[0], n_classes, kernel_size=1)

        if last_activation.casefold() == 'softmax':
            self.la = nn.Softmax()
        elif last_activation.casefold() == 'sigmoid':
            nn.init.xavier_uniform_(self.last.weight) # works well for sigmoid
            self.la = nn.Sigmoid()
        else:
            self.la = nn.Sequential()

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)    #1
                x = F.avg_pool3d(x, self.scale_factor)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1]) 
        x = self.last(x)
        x = self.la(x)
        return x
    
    
    
    def purge_model(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        gc.collect()
        pl.utilities.memory.garbage_collection_cuda()

    def training_step(self, batch, batch_nb):
        x = batch["img"]["data"]
        y = batch["label"]["data"]
        y_hat = self.forward(x)
        losses = self.compute_loss(y,y_hat)
        real_batch_size = x.shape[0]*torch.distributed.get_world_size() if torch.distributed.group.WORLD is not None else x.shape[0]
        loss = losses.pop('loss')
        self.log('Train/'+self.loss['name'], loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        self.log('Train/loss',               loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=real_batch_size)
        for name, value in losses.items():
            self.log('Train/'+name,  value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        evaluated_metrics = self.compute_additional_metrics(y,y_hat, 'training')        
        for name in evaluated_metrics:
            self.log('Train/'+name, evaluated_metrics[name], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        if False:
            if batch_nb == 0 and self.current_epoch!=0:
                i_image = x[0,0,:,:,x.shape[4]//2]         if self.dims==3 else x[0,0,:,:]
                o_image = y_hat[0,0,:,:,y_hat.shape[4]//2] if self.dims==3 else y_hat[0,0,:,:]
                self.loggers[0].log_image('Train-sample/input',  [wutils.convert_to_wandb_image((i_image-i_image.min())/(i_image.max()-i_image.min()))])
                self.loggers[0].log_image('Train-sample/output', [wutils.convert_to_wandb_image((o_image-o_image.min())/(o_image.max()-o_image.min()))])
        return loss if torch.isfinite(loss) else None
    
    def validation_step(self, batch, batch_nb):
        x = batch["img"]["data"]
        y = batch["label"]["data"]
        y_hat = self.forward(x)
        losses = self.compute_loss(y,y_hat)
        loss = losses.pop('loss')
        real_batch_size = x.shape[0]*torch.distributed.get_world_size(torch.distributed.group.WORLD) if torch.distributed.group.WORLD is not None else x.shape[0]
        self.log('Valid/'+self.loss['name'],  loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        self.log("Valid/loss",        loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=real_batch_size)
        for name, value in losses.items():
            self.log('Valid/val-'+name,  value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        evaluated_metrics = self.compute_additional_metrics(y,y_hat, 'validation')        
        for name in evaluated_metrics:
            self.log('Valid/'+name, evaluated_metrics[name], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        if False:
            if batch_nb == 0 and self.current_epoch!=0:
                i_image = x[0,0,:,:,x.shape[4]//2]         if self.dims==3 else x[0,0,:,:]
                o_image = y_hat[0,0,:,:,y_hat.shape[4]//2] if self.dims==3 else y_hat[0,0,:,:]
                self.loggers[0].log_image('Valid-sample/input',  [wutils.convert_to_wandb_image((i_image-i_image.min())/(i_image.max()-i_image.min()))])
                self.loggers[0].log_image('Valid-sample/output', [wutils.convert_to_wandb_image((o_image-o_image.min())/(o_image.max()-o_image.min()))])
        
        return loss
    
    def test_step(self, batch, batch_nb):
        x = batch["img"]["data"]
        y = batch["label"]["data"]
        y_hat = self.forward(x)

        if not hasattr(self, 'grid_sampler'): # FIXME : Careful, grid_sampler varies for each subject!
            self.set_test_grid_sampler(self.data_module.grid_sampler)
        locations = batch[tio.LOCATION]
        layered_batch = torch.cat((y, y_hat, x), dim=1)
        self.output_aggregator.add_batch(layered_batch, locations)
    
    def predict_step(self, batch, batch_nb):
        x = batch["img"]["data"]
        y_hat = self.forward(x)

        if not hasattr(self, 'grid_sampler'): # FIXME : Careful, grid_sampler varies for each subject!
            self.set_test_grid_sampler(self.data_module.grid_sampler)
        locations = batch[tio.LOCATION]
        self.output_aggregator.add_batch(y_hat, locations)
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    def on_test_epoch_end(self):
        output_tensor = self.output_aggregator.get_output_tensor()
        y     = output_tensor[0,:,:,:]
        y_hat = output_tensor[1,:,:,:] # is on RAM
        x     = output_tensor[2,:,:,:] # is on RAM
        del self.output_aggregator
        self.purge_model()
        if output_tensor.device != self.device:
            y_hat = y_hat.to(device=self.device)
            y     =     y.to(device=self.device)
            x     =     x.to(device=self.device)
        processing_args = self.data_module.test.get('post_processing')
        processed = common.hook_ex_external_code(processing_args=processing_args, y_hat=y_hat, y=y)
        y_hat = processed.get('y_hat',y_hat)
        y     = processed.get('y',y)
        metrics = self.compute_additional_metrics(y,y_hat, 'test', x=x)
        common.save_results_on_disk(self.data_module.test_and_predict['dataset'].get('results'), metrics)
        for name in metrics:
            self.log('Test/'+name, metrics[name]) 

    def on_predict_epoch_end(self):
        output_tensor = self.output_aggregator.get_output_tensor()
        y_hat = output_tensor[0,:,:,:]
        del output_tensor
        processing_args = self.data_module.predict.get('post_processing')
        processed = common.hook_ex_external_code(processing_args=processing_args, y_hat=y_hat)
        y_hat = processed.get('y_hat',y_hat)
        tmp = y_hat.cpu().detach().numpy() #if torch.is_tensor(output_tensor) else output_tensor
        tmp = np.squeeze(tmp)
        tmp = np.transpose(tmp,axes=(1,0,2))
        output_dir = self.data_module.test_and_predict['dataset'].get('results')
        if output_dir is not None:
            print('Saving output in '+output_dir)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
            for slice_number in range(tmp.shape[2]):
                im = Image.fromarray(tmp[:,:,slice_number])
                im.save(os.path.join(output_dir, 'infer_'+str(slice_number)+'.tif'))
        else:
            raise Exception('Output path for the prediction has not been set')
        
    def configure_optimizers(self):
        if self.optimizer.casefold() == 'adam':
            return torch.optim.Adam(self.parameters(), **self.optimizer_param)
        elif self.optimizer.casefold() == 'deepspeedcpuadam':
            return DeepSpeedCPUAdam(self.parameters(), **self.optimizer_param)
        elif self.optimizer.casefold() == 'fusedadam':
            return FusedAdam(self.parameters(), **self.optimizer_param)
        elif self.optimizer.casefold() == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), **self.optimizer_param) #weight_decay=1e-8 

    def set_test_grid_sampler(self, grid_sampler):
        self.grid_sampler = grid_sampler
        self.output_aggregator = tio.inference.GridAggregator(self.grid_sampler, overlap_mode='crop') #crop/average

    def set_dataloader(self, data_module):
        self.data_module = data_module
    
    def save_model(self):
        if self.log_path is not None:
            torch.save(self.state_dict(), os.path.join(self.log_path, self.name+".pt"))
        else:
            raise Exception("Cannot save model if log_path has not been defined.")

    def compute_additional_metrics(self, y, y_hat, stage, **additional_inputs):
        values = {}
        metrics = self.eval_metrics.get(stage,{})
        metrics = [metrics] if isinstance(metrics, dict) else metrics
        for metric in metrics:
            if metric != {}:
                values[metric['name']] = common.compute_metric(y, y_hat, metric, **additional_inputs)
        return values
    
    def compute_loss(self, y, y_hat, **additional_inputs):
        return common.compute_metric(y, y_hat, self.loss, **additional_inputs)
    
   
        

class UNetConvBlock(pl.LightningModule):
    def __init__(self, in_size, out_size, padding, batch_norm, activation,dropout=None,stride=1):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv3d(in_size, out_size, stride=stride, kernel_size=3, padding=padding, padding_mode='replicate'))
        if activation.casefold() == 'prelu':
            #nn.init.kaiming_normal_(block[-1].weight)
            block.append(nn.PReLU())
        elif activation.casefold() == 'leakyrelu':
            nn.init.kaiming_normal_(block[-1].weight)
            block.append(nn.LeakyReLU(0.01))
        elif activation.casefold() == 'sigmoid':
            #nn.init.xavier_uniform_(block[-1].weight) # works well for sigmoid
            block.append(nn.Sigmoid())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        block.append(nn.Conv3d(out_size, out_size, stride=stride, kernel_size=3, padding=padding, padding_mode='replicate'))
        if activation.casefold() == 'prelu':
            #nn.init.kaiming_normal_(block[-1].weight)
            block.append(nn.PReLU())
        elif activation.casefold() == 'leakyrelu':
            nn.init.kaiming_normal_(block[-1].weight)
            block.append(nn.LeakyReLU(0.01))
        elif activation.casefold() == 'sigmoid':
            #nn.init.xavier_uniform_(block[-1].weight) # works well for sigmoid
            block.append(nn.Sigmoid())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))
        if dropout is not None:
            block.append(nn.Dropout3d(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(pl.LightningModule):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, activation, scale_factor=2, dropout=None):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=scale_factor,
                                         stride=scale_factor)
        elif up_mode == 'upsample':
            #self.up = nn.Sequential(nn.Upsample(mode='trilinear', scale_factor=scale_factor, align_corners=False),
            self.up = nn.Sequential(nn.Upsample(mode='trilinear', scale_factor=scale_factor),
                                    nn.Conv3d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding=padding, batch_norm=batch_norm, activation=activation, dropout=dropout)

    def center_crop(self, layer, target_size):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_z = (layer_depth - target_size[0]) // 2
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        return layer[:, :, diff_z:(diff_z + target_size[0]), diff_y:(diff_y + target_size[1]), diff_x:(diff_x + target_size[2])]
        #  _, _, layer_height, layer_width = layer.size() #for 2D data
        # diff_y = (layer_height - target_size[0]) // 2
        # diff_x = (layer_width - target_size[1]) // 2 
        # return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        # bridge = self.center_crop(bridge, up.shape[2:]) #sending shape ignoring 2 digit, so target size start with 0,1,2
        #up = F.interpolate(up, size=bridge.shape[2:], mode='trilinear', align_corners=False) # Maybe in case the size differ between the bridge and the up-layers
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out

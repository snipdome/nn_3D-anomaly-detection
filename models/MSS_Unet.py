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

class MSS_Unet(pl.LightningModule):
     
    def __init__(self, name, checkpoint_path, log_path=None, n_channels=1, n_classes=1, depth=3, wf=6, padding=True, 
                 batch_norm=False, up_mode='upsample', activation='LeakyReLu', loss={'name': 'cross_entropy'}, evaluate_metrics={}, dropout=None, last_activation='', optimizer_parameters=None, optimizer = 'RMSprop',**kwargs):
        super(MSS_Unet, self).__init__()
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
        prev_channels = n_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i), padding=padding, batch_norm=batch_norm, activation=activation, dropout=dropout))
            prev_channels = 2**(wf+i)

        self.up_path       = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode=up_mode,
                                            padding=padding, batch_norm= batch_norm, activation=activation, dropout=dropout))
            prev_channels = 2**(wf+i)

        self.last = UNetConvBlock(prev_channels, 1, padding=padding, activation=last_activation, is_single_conv=True)

        self.up_path_exits = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            prev_channels = 2**(wf+i)
            if i != 0: 
                self.up_path_exits.append(UNetConvBlock(prev_channels, 1,  padding=padding, activation=[activation,last_activation]))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)    #1
                x = F.avg_pool3d(x, 2)

        y = []
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1]) 
            if i<self.depth-2:
                y.append(self.up_path_exits[i](x))

        x = self.last(x) #FIXME: last layer is not different from others: it should not have its own last_layer
        y.append(x) 
        return y # from deeper to outer

    def purge_model(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        gc.collect()
        pl.utilities.memory.garbage_collection_cuda()

    def training_step(self, batch, batch_nb):
        x = batch["img"]["data"]
        y = batch["label"]["data"]
        y_size = y.shape[2:]
        y_hat = self.forward(x)
        loss = []
        loss_weight_factor = []
        for i, y_hat_layer in enumerate(y_hat):
            loss_weight_factor.append(1/(len(y_hat)-i))
            if i < self.depth-1:
                interp_y_hat = torch.nn.functional.interpolate(input=y_hat_layer, size=y_size)
                losses = self.compute_loss(y, interp_y_hat)
            else: # is last layer
                losses = self.compute_loss(y, y_hat_layer)
            loss.append(loss_weight_factor[i] * losses.pop('loss'))
        red_loss = sum(loss)/sum(loss_weight_factor)
            
        real_batch_size = x.shape[0]*torch.distributed.get_world_size(torch.distributed.group.WORLD) if torch.distributed.group.WORLD is not None else x.shape[0]
        self.log('Train/focal_tversky_loss', loss[-1], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('Train/loss',       red_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        evaluated_metrics = self.compute_additional_metrics(y,y_hat[-1], 'training')
        for name in evaluated_metrics:
            self.log('Train/'+name, evaluated_metrics[name], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        if False:
            if batch_nb == 0 and self.current_epoch!=0:
                i_image = x[0,0,:,:,x.shape[4]//2]         if self.dims==3 else x[0,0,:,:]
                o_image = y_hat[0,0,:,:,y_hat.shape[4]//2] if self.dims==3 else y_hat[0,0,:,:]
                self.loggers[0].log_image('Train-sample/input',  [wutils.convert_to_wandb_image((i_image-i_image.min())/(i_image.max()-i_image.min()))])
                self.loggers[0].log_image('Train-sample/output', [wutils.convert_to_wandb_image((o_image-o_image.min())/(o_image.max()-o_image.min()))])
        return red_loss if torch.isfinite(red_loss) else None

    def validation_step(self, batch, batch_nb):
        x = batch["img"]["data"]
        y = batch["label"]["data"]
        y_size = y.shape[2:]
        y_hat = self.forward(x)
        # Calculate group loss and the Focal Tversky's
        val_loss = []
        loss_weight_factor = []
        for i, y_hat_layer in enumerate(y_hat):
            #loss_weight_factor.append(1/(2**(len(y_hat)-1-i)))
            loss_weight_factor.append(1/(len(y_hat)-i))
            if i < self.depth-1:
                interp_y_hat = torch.nn.functional.interpolate(input=y_hat_layer, size=y_size)
                losses = self.compute_loss(y, interp_y_hat)
            else: # is last layer
                losses = self.compute_loss(y, y_hat_layer)
            val_loss.append(loss_weight_factor[i] * losses.pop('loss'))
        val_red_loss = sum(val_loss)/sum(loss_weight_factor)
        real_batch_size = x.shape[0]*torch.distributed.get_world_size(torch.distributed.group.WORLD) if torch.distributed.group.WORLD is not None else x.shape[0]
        self.log("Valid/loss",             val_red_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        self.log("Valid/focal_tversky_loss", val_loss[-1], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        evaluated_metrics = self.compute_additional_metrics(y,y_hat[-1], 'validation')
        for name in evaluated_metrics:
            self.log('Valid/'+name, evaluated_metrics[name], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
        if False:
            if batch_nb == 0 and self.current_epoch!=0:
                i_image = x[0,0,:,:,x.shape[4]//2]         if self.dims==3 else x[0,0,:,:]
                o_image = y_hat[0,0,:,:,y_hat.shape[4]//2] if self.dims==3 else y_hat[0,0,:,:]
                self.loggers[0].log_image('Valid-sample/input',  [wutils.convert_to_wandb_image((i_image-i_image.min())/(i_image.max()-i_image.min()))])
                self.loggers[0].log_image('Valid-sample/output', [wutils.convert_to_wandb_image((o_image-o_image.min())/(o_image.max()-o_image.min()))])
        return val_red_loss
    
    def test_step(self, batch, batch_nb):
        x = batch["img"]["data"]
        y = batch["label"]["data"]
        y_hat = self.forward(x)

        if not hasattr(self, 'grid_sampler'): # FIXME : Careful, grid_sampler varies for each subject!
            self.set_test_grid_sampler(self.data_module.grid_sampler)
        locations = batch[tio.LOCATION]
        layered_batch = torch.cat((y, y_hat[-1], x), dim=1)
        self.output_aggregator.add_batch(layered_batch, locations)

    def predict_step(self, batch, batch_nb):
        x = batch["img"]["data"]
        y_hat = self.forward(x)

        if not hasattr(self, 'grid_sampler'): # FIXME : Careful, grid_sampler varies for each subject!
            self.set_test_grid_sampler(self.data_module.grid_sampler)
        locations = batch[tio.LOCATION]
        self.output_aggregator.add_batch(y_hat[-1], locations)

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
    def __init__(self, in_size, out_size, padding=1, batch_norm=False, activation='prelu', dropout=None, is_single_conv=False):
        super(UNetConvBlock, self).__init__()
        if len(activation)==2:
            activation1, activation2 = activation[0].casefold(), activation[1].casefold()
        else:
            activation1 = activation2 = activation.casefold() 

        block = []
        block.append(nn.Conv3d(in_size, out_size, kernel_size=3, padding=padding, padding_mode='replicate'))
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
            block.append(nn.Conv3d(out_size, out_size, kernel_size=3, padding=padding, padding_mode='replicate'))
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
            if dropout is not None:
                block.append(nn.Dropout3d(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(pl.LightningModule):
    def __init__(self, in_size, out_size, up_mode, batch_norm, activation, padding=1, dropout=None):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='trilinear', scale_factor=2, align_corners=False),
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

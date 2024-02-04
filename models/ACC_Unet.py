# 
# This file is part of the nn_3D-anomaly-detection distribution (https://github.com/snipdome/nn_3D-anomaly-detection).
# Copyright (c) 2022-2023 imec-Vision Lab, University of Antwerp
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
#
# This file incorporates work covered originally by the "ACC-UNet" package
# (https://github.com/kiharalab/ACC-UNet), which is licensed as follows:
#
#   This program is free software: you can redistribute it and/or modify  
#   it under the terms of the GNU General Public License as published by  
#   the Free Software Foundation, version 3.
#
#   This program is distributed in the hope that it will be useful, but 
#   WITHOUT ANY WARRANTY; without even the implied warranty of 
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
#   General Public License for more details.
#
#   You should have received a copy of the GNU General Public License 
#   along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# The original content has been modified:
#
# 05 Feb 2024 - imec-Vision Lab, University of Antwerp: Added support for 3D data

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

    
class ACC_Unet(pl.LightningModule):
     
    def __init__(self, name, checkpoint_path, log_path=None, n_channels=1, n_classes=1, depth=3, n_filts=32, channel_depths=None, padding=True, 
                 batch_norm=False, up_mode='upsample', insta_norm=None, scale_factor=2, stride=1, input_size=256, activation='LeakyReLu',loss={'name': 'cross_entropy'}, evaluate_metrics={}, dropout=False, last_activation='', optimizer_parameters=None, optimizer = 'RMSprop',**kwargs):
        '''
        n_filts : multiplier of the number of filters throughout the model. Increase this to make the model wider. Decrease this to make the model ligher. Defaults to 32.
        
        '''
        
        super(ACC_Unet, self).__init__()
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
     
        assert n_classes == 1, 'n classes different from 1 not implemented yet'

        self.pool = torch.nn.MaxPool2d(2) if self.dims==2 else torch.nn.MaxPool3d(2)

        self.cnv11 = HANCBlock(n_channels, n_filts, k=3, inv_fctr=3, dims=self.dims)
        self.cnv12 = HANCBlock(n_filts, n_filts, k=3, inv_fctr=3, dims=self.dims)

        self.cnv21 = HANCBlock(n_filts, n_filts * 2, k=3, inv_fctr=3, dims=self.dims)
        self.cnv22 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3, dims=self.dims)

        self.cnv31 = HANCBlock(n_filts * 2, n_filts * 4, k=3, inv_fctr=3, dims=self.dims)
        self.cnv32 = HANCBlock(n_filts * 4, n_filts * 4, k=3, inv_fctr=3, dims=self.dims)

        self.cnv41 = HANCBlock(n_filts * 4, n_filts * 8, k=2, inv_fctr=3, dims=self.dims)
        self.cnv42 = HANCBlock(n_filts * 8, n_filts * 8, k=2, inv_fctr=3, dims=self.dims)

        self.cnv51 = HANCBlock(n_filts * 8, n_filts * 16, k=1, inv_fctr=3, dims=self.dims)
        self.cnv52 = HANCBlock(n_filts * 16, n_filts * 16, k=1, inv_fctr=3, dims=self.dims)

        self.rspth1 = ResPath(n_filts, 4, dims=self.dims)
        self.rspth2 = ResPath(n_filts * 2, 3, dims=self.dims)
        self.rspth3 = ResPath(n_filts * 4, 2, dims=self.dims)
        self.rspth4 = ResPath(n_filts * 8, 1, dims=self.dims)

        self.mlfc1 = MLFC(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, lenn=1, dims=self.dims)
        self.mlfc2 = MLFC(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, lenn=1, dims=self.dims)
        self.mlfc3 = MLFC(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, lenn=1, dims=self.dims)

        self.up6 = torch.nn.ConvTranspose2d(n_filts * 16, n_filts * 8, kernel_size=(2, 2), stride=2) if self.dims==2 else torch.nn.ConvTranspose3d(n_filts * 16, n_filts * 8, kernel_size=(2, 2, 2), stride=2)
        self.cnv61 = HANCBlock(n_filts * 8 + n_filts * 8, n_filts * 8, k=2, inv_fctr=3, dims=self.dims)
        self.cnv62 = HANCBlock(n_filts * 8, n_filts * 8, k=2, inv_fctr=3, dims=self.dims)

        self.up7 = torch.nn.ConvTranspose2d(n_filts * 8, n_filts * 4, kernel_size=(2, 2), stride=2) if self.dims==2 else torch.nn.ConvTranspose3d(n_filts * 8, n_filts * 4, kernel_size=(2, 2, 2), stride=2)
        self.cnv71 = HANCBlock(n_filts * 4 + n_filts * 4, n_filts * 4, k=3, inv_fctr=3, dims=self.dims)
        self.cnv72 = HANCBlock(n_filts * 4, n_filts * 4, k=3, inv_fctr=34, dims=self.dims)
        self.out7 = OutLayer(n_filts * 4, n_classes, dims=self.dims)

        self.up8 = torch.nn.ConvTranspose2d(n_filts * 4, n_filts * 2, kernel_size=(2, 2), stride=2) if self.dims==2 else torch.nn.ConvTranspose3d(n_filts * 4, n_filts * 2, kernel_size=(2, 2, 2), stride=2)
        self.cnv81 = HANCBlock(n_filts * 2 + n_filts * 2, n_filts * 2, k=3, inv_fctr=3, dims=self.dims)
        self.cnv82 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3, dims=self.dims)
        self.out8 = OutLayer(n_filts * 2, n_classes, dims=self.dims)

        self.up9 = torch.nn.ConvTranspose2d(n_filts * 2, n_filts, kernel_size=(2, 2), stride=2) if self.dims==2 else torch.nn.ConvTranspose3d(n_filts * 2, n_filts, kernel_size=(2, 2, 2), stride=2)
        self.cnv91 = HANCBlock(n_filts + n_filts, n_filts, k=3, inv_fctr=3, dims=self.dims)
        self.cnv92 = HANCBlock(n_filts, n_filts, k=3, inv_fctr=3, dims=self.dims)
        self.out9 = OutLayer(n_filts, n_classes, dims=self.dims)

            

    def forward(self, x):
        x1 = x
        output = []

        x2 = self.cnv11(x1)
        x2 = self.cnv12(x2)

        x2p = self.pool(x2)

        x3 = self.cnv21(x2p)
        x3 = self.cnv22(x3)

        x3p = self.pool(x3)

        x4 = self.cnv31(x3p)
        x4 = self.cnv32(x4)

        x4p = self.pool(x4)

        x5 = self.cnv41(x4p)
        x5 = self.cnv42(x5)

        x5p = self.pool(x5)

        x6 = self.cnv51(x5p)
        x6 = self.cnv52(x6)

        x2 = self.rspth1(x2)
        x3 = self.rspth2(x3)
        x4 = self.rspth3(x4)
        x5 = self.rspth4(x5)

        x2, x3, x4, x5 = self.mlfc1(x2, x3, x4, x5)
        x2, x3, x4, x5 = self.mlfc2(x2, x3, x4, x5)
        x2, x3, x4, x5 = self.mlfc3(x2, x3, x4, x5)

        x7 = self.up6(x6)
        x7 = self.cnv61(torch.cat([x7, x5], dim=1))
        x7 = self.cnv62(x7)

        x8 = self.up7(x7)
        x8 = self.cnv71(torch.cat([x8, x4], dim=1))
        x8 = self.cnv72(x8)
        output.append(self.out7(x8))

        x9 = self.up8(x8)
        x9 = self.cnv81(torch.cat([x9, x3], dim=1))
        x9 = self.cnv82(x9)
        output.append(self.out8(x9))

        x10 = self.up9(x9)
        x10 = self.cnv91(torch.cat([x10, x2], dim=1))
        x10 = self.cnv92(x10)
        output.append(self.out9(x10))

        return output
    
    
    
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
                o_image = y_hat[-1][0,0,:,:,y_hat[-1].shape[4]//2] if self.dims==3 else y_hat[-1][0,0,:,:]
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
    
   
################################## HELPER CLASSES ##################################


class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels, dims=2):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
            dims (int): number of dimensions (default: {2})
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1) if dims==2 else torch.nn.AdaptiveAvgPool3d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels) if dims==2 else torch.nn.BatchNorm3d(num_channels)
        self.forward = self.forward_2d if dims==2 else self.forward_3d


    def forward_2d(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out
    
    def forward_3d(self, inp):
        batch_size, num_channels, H, W, D = inp.size()
        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))
        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1, 1))
        out = self.bn(out)
        out = self.act(out)
        return out



class HANCLayer(torch.nn.Module):
    """
    Implements Hierarchical Aggregation of Neighborhood Context operation
    """

    def __init__(self, in_chnl, out_chnl, k, dims=2):
        """
        Initialization

        Args:
            in_chnl (int): number of input channels
            out_chnl (int): number of output channels
            k (int): value of k in HANC
            dims (int): number of dimensions (default: {2})
        """

        super(HANCLayer, self).__init__()

        self.k = k

        self.cnv = torch.nn.Conv2d((2 * k - 1) * in_chnl, out_chnl, kernel_size=(1, 1)) if dims==2 else torch.nn.Conv3d((2 * k - 1) * in_chnl, out_chnl, kernel_size=(1, 1, 1))
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl) if dims==2 else torch.nn.BatchNorm3d(out_chnl)
        self.forward = self.forward_2d if dims==2 else self.forward_3d

    def forward_2d(self, inp):

        batch_size, num_channels, H, W = inp.size()

        x = inp

        if self.k == 1:
            x = inp

        elif self.k == 2:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                ],
                dim=2,
            )

        elif self.k == 3:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                ],
                dim=2,
            )

        elif self.k == 4:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.AvgPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.MaxPool2d(8)(x)),
                ],
                dim=2,
            )

        elif self.k == 5:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.AvgPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=16)(torch.nn.AvgPool2d(16)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.MaxPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=16)(torch.nn.MaxPool2d(16)(x)),
                ],
                dim=2,
            )

        x = x.view(batch_size, num_channels * (2 * self.k - 1), H, W, D)

        x = self.act(self.bn(self.cnv(x)))

        return x
    
    def forward_3d(self, inp):
            
            batch_size, num_channels, H, W, D = inp.size()
    
            x = inp
    
            if self.k == 1:
                x = inp
    
            elif self.k == 2:
                x = torch.concat(
                    [
                        x,
                        torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool3d(2)(x)),
                        torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool3d(2)(x)),
                    ],
                    dim=2,
                )
    
            elif self.k == 3:
                x = torch.concat(
                    [
                        x,
                        torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool3d(2)(x)),
                        torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool3d(4)(x)),
                        torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool3d(2)(x)),
                        torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool3d(4)(x)),
                    ],
                    dim=2,
                )
    
            elif self.k == 4:
                x = torch.concat(
                    [
                        x,
                        torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool3d(2)(x)),
                        torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool3d(4)(x)),
                        torch.nn.Upsample(scale_factor=8)(torch.nn.AvgPool3d(8)(x)),
                        torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool3d(2)(x)),
                        torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool3d(4)(x)),
                        torch.nn.Upsample(scale_factor=8)(torch.nn.MaxPool3d(8)(x)),
                    ],
                    dim=2,
                )
    
            elif self.k == 5:
                x = torch.concat(
                    [
                        x,
                        torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool3d(2)(x)),
                        torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool3d(4)(x)),
                        torch.nn.Upsample(scale_factor=8)(torch.nn.AvgPool3d(8)(x)),
                        torch.nn.Upsample(scale_factor=16)(torch.nn.AvgPool3d(16)(x)),
                        torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool3d(2)(x)),
                        torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool3d(4)(x)),
                        torch.nn.Upsample(scale_factor=8)(torch.nn.MaxPool3d(8)(x)),
                        torch.nn.Upsample(scale_factor=16)(torch.nn.MaxPool3d(16)(x)),
                    ],
                    dim=2,
                )
            
            x = x.view(batch_size, num_channels * (2 * self.k - 1), H, W, D)
            x = self.act(self.bn(self.cnv(x)))
            return x



class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.sqe(self.activation(x))

class Conv3d_batchnorm(torch.nn.Module):
    """
    3D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1, 1),
        activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv3d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm3d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters, dims=3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.sqe(self.activation(x))

class Conv2d_channel(torch.nn.Module):
    """
    2D pointwise Convolutional layers        
    """

    def __init__(self, num_in_filters, num_out_filters):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))

class Conv3d_channel(torch.nn.Module):
    """
    3D pointwise Convolutional layers        
    """

    def __init__(self, num_in_filters, num_out_filters):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv3d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm3d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters, dims=3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))

class HANCBlock(torch.nn.Module):
    """
    Encapsulates HANC block
    """

    def __init__(self, n_filts, out_channels, k=3, inv_fctr=3, dims=2):
        """
        Initialization

        Args:
            n_filts (int): number of filters
            out_channels (int): number of output channel
            activation (str, optional): activation function. Defaults to 'LeakyReLU'.
            k (int, optional): k in HANC. Defaults to 1.
            inv_fctr (int, optional): inv_fctr in HANC. Defaults to 4.
        """

        super().__init__()

        self.conv1 = torch.nn.Conv2d(n_filts, n_filts * inv_fctr, kernel_size=1) if dims==2 else torch.nn.Conv3d(n_filts, n_filts * inv_fctr, kernel_size=1)
        self.norm1 = torch.nn.BatchNorm2d(n_filts * inv_fctr) if dims==2 else torch.nn.BatchNorm3d(n_filts * inv_fctr)

        self.conv2 = torch.nn.Conv2d(
            n_filts * inv_fctr,
            n_filts * inv_fctr,
            kernel_size=3,
            padding=1,
            groups=n_filts * inv_fctr,
        ) if dims==2 else torch.nn.Conv3d(
            n_filts * inv_fctr,
            n_filts * inv_fctr,
            kernel_size=3,
            padding=1,
            groups=n_filts * inv_fctr,
        )
        self.norm2 = torch.nn.BatchNorm2d(n_filts * inv_fctr) if dims==2 else torch.nn.BatchNorm3d(n_filts * inv_fctr)

        self.hnc = HANCLayer(n_filts * inv_fctr, n_filts, k, dims) if dims==2 else HANCLayer(n_filts * inv_fctr, n_filts, k, dims)

        self.norm = torch.nn.BatchNorm2d(n_filts) if dims==2 else torch.nn.BatchNorm3d(n_filts)

        self.conv3 = torch.nn.Conv2d(n_filts, out_channels, kernel_size=1) if dims==2 else torch.nn.Conv3d(n_filts, out_channels, kernel_size=1)
        self.norm3 = torch.nn.BatchNorm2d(out_channels) if dims==2 else torch.nn.BatchNorm3d(out_channels)

        self.sqe = ChannelSELayer(out_channels, dims=dims)

        self.activation = torch.nn.LeakyReLU()


    def forward(self, inp):

        x = self.conv1(inp)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.hnc(x)

        x = self.norm(x + inp)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.sqe(x)

        return x



class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl, dims=2):
        """
        Initialization

        Args:
            in_chnls (int): number of input channels
            n_lvl (int): number of blocks or levels
        """

        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])

        self.bn = torch.nn.BatchNorm2d(in_chnls) if dims==2 else torch.nn.BatchNorm3d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls) if dims==2 else torch.nn.BatchNorm3d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls, kernel_size=(3, 3), padding=1) if dims==2 else torch.nn.Conv3d(in_chnls, in_chnls, kernel_size=(3, 3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls) if dims==2 else torch.nn.BatchNorm3d(in_chnls))
            self.sqes.append(ChannelSELayer(in_chnls, dims=dims))   


    def forward(self, x):

        for i in range(len(self.convs)):
            x = x + self.sqes[i](self.act(self.bns[i](self.convs[i](x))))

        return self.sqe(self.act(self.bn(x)))



class MLFC(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1, dims=2):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
            in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2) if dims==2 else torch.nn.AvgPool3d(2)  # used for upsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])


        if dims==2:
            for i in range(lenn):
                self.cnv_blks1.append(
                    Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
                )
                self.cnv_mrg1.append(Conv2d_batchnorm(2 * in_filters1, in_filters1, (1, 1)))
                self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
                self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

                self.cnv_blks2.append(
                    Conv2d_batchnorm(self.in_filters, in_filters2, (1, 1))
                )
                self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
                self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
                self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

                self.cnv_blks3.append(
                    Conv2d_batchnorm(self.in_filters, in_filters3, (1, 1))
                )
                self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
                self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
                self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

                self.cnv_blks4.append(
                    Conv2d_batchnorm(self.in_filters, in_filters4, (1, 1))
                )
                self.cnv_mrg4.append(Conv2d_batchnorm(2 * in_filters4, in_filters4, (1, 1)))
                self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
                self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))
        else:
            for i in range(lenn):
                self.cnv_blks1.append(
                    Conv3d_batchnorm(self.in_filters, in_filters1, (1, 1, 1))
                )
                self.cnv_mrg1.append(Conv3d_batchnorm(2 * in_filters1, in_filters1, (1, 1, 1)))
                self.bns1.append(torch.nn.BatchNorm3d(in_filters1))
                self.bns_mrg1.append(torch.nn.BatchNorm3d(in_filters1))

                self.cnv_blks2.append(
                    Conv3d_batchnorm(self.in_filters, in_filters2, (1, 1, 1))
                )
                self.cnv_mrg2.append(Conv3d_batchnorm(2 * in_filters2, in_filters2, (1, 1, 1)))
                self.bns2.append(torch.nn.BatchNorm3d(in_filters2))
                self.bns_mrg2.append(torch.nn.BatchNorm3d(in_filters2))

                self.cnv_blks3.append(
                    Conv3d_batchnorm(self.in_filters, in_filters3, (1, 1, 1))
                )
                self.cnv_mrg3.append(Conv3d_batchnorm(2 * in_filters3, in_filters3, (1, 1, 1)))
                self.bns3.append(torch.nn.BatchNorm3d(in_filters3))
                self.bns_mrg3.append(torch.nn.BatchNorm3d(in_filters3))

                self.cnv_blks4.append(
                    Conv3d_batchnorm(self.in_filters, in_filters4, (1, 1, 1))
                )
                self.cnv_mrg4.append(Conv3d_batchnorm(2 * in_filters4, in_filters4, (1, 1, 1)))
                self.bns4.append(torch.nn.BatchNorm3d(in_filters4))
                self.bns_mrg4.append(torch.nn.BatchNorm3d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1, dims=dims)
        self.sqe2 = ChannelSELayer(in_filters2, dims=dims) 
        self.sqe3 = ChannelSELayer(in_filters3, dims=dims)
        self.sqe4 = ChannelSELayer(in_filters4, dims=dims)

        self.forward = self.forward_2d if dims==2 else self.forward_3d


    def forward_2d(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                x1,
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                self.no_param_down(x1),
                                (x2),
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(x1)),
                                self.no_param_down(x2),
                                (x3),
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                                self.no_param_down(self.no_param_down(x2)),
                                self.no_param_down(x3),
                                x4,
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1], dim=2).view(batch_size, 2 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2], dim=2).view(batch_size, 2 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3], dim=2).view(batch_size, 2 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4], dim=2).view(batch_size, 2 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4

    def forward_3d(self, x1, x2, x3, x4):
            
            batch_size, _, h1, w1, d1 = x1.shape
            _, _, h2, w2, d2 = x2.shape
            _, _, h3, w3, d3 = x3.shape
            _, _, h4, w4, d4 = x4.shape
    
            for i in range(len(self.cnv_blks1)):
                x_c1 = self.act(
                    self.bns1[i](
                        self.cnv_blks1[i](
                            torch.cat(
                                [
                                    x1,
                                    self.no_param_up(x2),
                                    self.no_param_up(self.no_param_up(x3)),
                                    self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                                ],
                                dim=1,
                            )
                        )
                    )
                )
                x_c2 = self.act(
                    self.bns2[i](
                        self.cnv_blks2[i](
                            torch.cat(
                                [
                                    self.no_param_down(x1),
                                    (x2),
                                    (self.no_param_up(x3)),
                                    (self.no_param_up(self.no_param_up(x4))),
                                ],
                                dim=1,
                            )
                        )
                    )
                )
                x_c3 = self.act(
                    self.bns3[i](
                        self.cnv_blks3[i](
                            torch.cat(
                                [
                                    self.no_param_down(self.no_param_down(x1)),
                                    self.no_param_down(x2),
                                    (x3),
                                    (self.no_param_up(x4)),
                                ],
                                dim=1,
                            )
                        )
                    )
                )
                x_c4 = self.act(
                    self.bns4[i](
                        self.cnv_blks4[i](
                            torch.cat(
                                [
                                    self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                                    self.no_param_down(self.no_param_down(x2)),
                                    self.no_param_down(x3),
                                    x4,
                                ],
                                dim=1,
                            )
                        )
                    )
                )
    
                x_c1 = self.act(
                    self.bns_mrg1[i](
                        self.cnv_mrg1[i](
                            torch.cat([x_c1, x1], dim=2).view(batch_size, 2 * self.in_filters1, h1, w1, d1)
                        )
                        + x1
                    )
                )
                x_c2 = self.act(
                    self.bns_mrg2[i](
                        self.cnv_mrg2[i](
                            torch.cat([x_c2, x2], dim=2).view(batch_size, 2 * self.in_filters2, h2, w2, d2)
                        )
                        + x2
                    )
                )
                x_c3 = self.act(
                    self.bns_mrg3[i](
                        self.cnv_mrg3[i](
                            torch.cat([x_c3, x3], dim=2).view(batch_size, 2 * self.in_filters3, h3, w3, d3)
                        )
                        + x3
                    )
                )
                x_c4 = self.act(
                    self.bns_mrg4[i](
                        self.cnv_mrg4[i](
                            torch.cat([x_c4, x4], dim=2).view(batch_size, 2 * self.in_filters4, h4, w4, d4)
                        )
                        + x4
                    )
                )

            x1 = self.sqe1(x_c1)
            x2 = self.sqe2(x_c2)
            x3 = self.sqe3(x_c3)
            x4 = self.sqe4(x_c4)

            return x1, x2, x3, x4
    

class OutLayer(torch.nn.Module):

    def __init__(self, n_filts, n_classes, dims=2):
        super().__init__()
        self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1)) if dims==2 else torch.nn.Conv3d(n_filts, n_classes, kernel_size=(1, 1, 1))
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.out(x))
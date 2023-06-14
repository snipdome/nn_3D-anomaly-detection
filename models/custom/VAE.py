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


import os, pathlib, gc, time, numpy as np, nibabel as nib, matplotlib.pyplot as plt, PIL.Image as Image

import torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist
import pytorch_lightning as pl, torchio as tio, wandb
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from utils.stats import *
import utils.wandb as wutils
from utils.extern import *
from ..helpers import common_operations as common
from models.helpers.ce import *
from models.helpers.custom_layers import *

# https://arxiv.org/pdf/2109.06540.pdf



class ConvDownBlock3D(pl.LightningModule):
	def __init__(self, in_size, out_size, down_mode='avg', activation='Relu', bias=True, insta_norm=None, stride=1, batch_norm=None, padding=1, dropout=None, scale_factor=2, is_single_conv=False, flat_latent=True, kernel_size=3, padding_mode='replicate'):
		super(ConvDownBlock3D, self).__init__()
		
		self.down = nn.Sequential()
		self.down.append(nn.Conv3d(in_size, out_size, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=stride, bias=bias))

		if   activation == 'relu':
			nn.init.kaiming_normal_(self.down[-1].weight, nonlinearity='relu')
			self.down.append(nn.ReLU())
		if   activation == 'prelu':
			nn.init.kaiming_normal_(self.down[-1].weight, nonlinearity='leaky_relu')
			self.down.append(nn.PReLU())
		elif activation == 'leakyrelu':
			nn.init.kaiming_normal_(self.down[-1].weight, nonlinearity='leaky_relu')
			self.down.append(nn.LeakyReLU(0.01))
		elif activation == 'sigmoid':
			nn.init.xavier_normal_(self.down[-1].weight)
			self.down.append(nn.Sigmoid())
		if batch_norm:
			self.down.append(nn.BatchNorm3d(out_size))
		if insta_norm:
			self.down.append(nn.InstanceNorm3d(out_size))
		if dropout:
			self.down.append(nn.Dropout3d(p=dropout))

		if scale_factor != 1:
			if down_mode in ('average','avg'):
				self.down.append(nn.AvgPool3d(scale_factor))
			elif 'max':
				self.down.append(nn.MaxPool3d(scale_factor))
			else:
				raise Exception('The requested down_mode has not been implemented')

	def forward(self, x):
		return self.down(x)  
	
class ConvUpBlock3D(pl.LightningModule):
	def __init__(self, in_size, out_size, up_mode='upconv', activation='Relu', batch_norm=None, insta_norm=None,  dropout=None, scale_factor=2, is_single_conv=False, bridge_size=None, flat_latent=True, kernel_size=3, bias=True, stride=1, padding=1, **kwargs):
		super(ConvUpBlock3D, self).__init__()
		self.up= nn.Sequential()
		
		if up_mode == 'upconv':
			self.up.append(nn.ConvTranspose3d(in_size, out_size, kernel_size=kernel_size, stride=stride,  bias=bias, padding=0))
		elif up_mode == 'upsample':
			self.up.append(nn.Upsample(mode='trilinear', scale_factor=scale_factor, align_corners=False))
			self.up.append(nn.Conv3d(in_size, out_size, kernel_size=1, padding=padding, bias=bias))
		else:
			raise Exception('The requested up_mode has not been implemented')
			
		self.up.append(ConvDownBlock3D(out_size, out_size, padding=padding, kernel_size=3, batch_norm=batch_norm, activation=activation, dropout=dropout, scale_factor=1))


	def forward(self, x):
		return self.up(x)
	
class ConvDownBlock2D(pl.LightningModule):
	def __init__(self, in_size, out_size, down_mode='avg', activation='Relu', bias=True, insta_norm=None, stride=1, batch_norm=None, padding=1, dropout=None, scale_factor=2, is_single_conv=False, flat_latent=True, kernel_size=3, padding_mode='zeros'):
		super(ConvDownBlock2D, self).__init__()
		
		self.down = nn.Sequential()
		self.down.append(nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, stride=stride, bias=bias))

		if   activation == 'relu':
			nn.init.kaiming_normal_(self.down[-1].weight, nonlinearity='relu')
			self.down.append(nn.ReLU())
		if   activation == 'prelu':
			nn.init.kaiming_normal_(self.down[-1].weight, nonlinearity='leaky_relu')
			self.down.append(nn.PReLU())
		elif activation == 'leakyrelu':
			nn.init.kaiming_normal_(self.down[-1].weight, nonlinearity='leaky_relu')
			self.down.append(nn.LeakyReLU(0.01))
		elif activation == 'sigmoid':
			nn.init.xavier_normal_(self.down[-1].weight)
			self.down.append(nn.Sigmoid())
		if batch_norm:
			self.down.append(nn.BatchNorm2d(out_size))
		if insta_norm:
			self.down.append(nn.InstanceNorm2d(out_size))
		if dropout:
			self.down.append(nn.Dropout2d(p=dropout))

		if scale_factor != 1:
			if down_mode in ('average','avg'):
				self.down.append(nn.AvgPool2d(scale_factor))
			elif 'max':
				self.down.append(nn.MaxPool2d(scale_factor))
			else:
				raise Exception('The requested down_mode has not been implemented')

	def forward(self, x):
		return self.down(x)
	
class ConvUpBlock2D(pl.LightningModule):
	def __init__(self, in_size, out_size, up_mode='upconv', activation='Relu', batch_norm=None, insta_norm=None, padding=1, dropout=None, scale_factor=2, is_single_conv=False, bridge_size=None, flat_latent=True, kernel_size=3, bias=True, stride=1):
		super(ConvUpBlock2D, self).__init__()
		self.up= nn.Sequential()
		
		if up_mode == 'upconv':
			self.up.append(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=0, bias=bias))
		elif up_mode == 'upsample':
			self.up.append(nn.Upsample(mode='bilinear', scale_factor=scale_factor, align_corners=False))
			self.up.append(nn.Conv2d(in_size, out_size, kernel_size=1, padding=padding, bias=bias))
		else:
			raise Exception('The requested up_mode has not been implemented')
		
		self.up.append(ConvDownBlock2D(out_size, out_size, kernel_size=3,padding=padding, batch_norm=batch_norm, activation=activation, dropout=dropout, scale_factor=1))

	def forward(self, x):
		return self.up(x)

# Based on: Context-encoding Variational Autoencoder for Unsupervised Anomaly Detection
class VAE(pl.LightningModule):
	def __init__(
		self, name, checkpoint_path=None, log_path=None, n_channels=1, n_classes=1, depth=3, wf=6, padding=True, input_size=256, patch_size=64, channel_depths=1024, flat_latent=True,
		batch_norm=None, insta_norm=None, up_mode='upsample', activation='LeakyReLu', loss={'name': 'cross_entropy'},  kernel_size=4,
		evaluate_metrics={}, dropout=None, last_activation='sigmoid', optimizer_parameters=None, optimizer = 'RMSprop', **kwargs
	):
		super(VAE, self).__init__()
		self.save_hyperparameters()
		self.name = name
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.checkpoint_path = checkpoint_path
		self.padding = padding
		self.log_path = log_path
		self.depth = depth
		self.activation = activation
		self.last_activation = last_activation
		self.input_size = input_size
		self.optimizer = optimizer
		self.optimizer_param = optimizer_parameters
		self.loss = loss
		self.eval_metrics = evaluate_metrics
		self.kernel_size = kernel_size
		self.log_wandb_images = kwargs.get('log_wandb_images', False)
			  
		if not isinstance(input_size, list):
			self.dims = 3
		elif len(input_size) == 2:
			self.dims = 2
			assert (input_size[0]==input_size[1]), 'Wrong dimensions in input'
			input_size = input_size[0]
		elif len(input_size) == 3:
			assert (input_size[0]==input_size[1] and input_size[0]==input_size[2]), 'Wrong dimensions in input'
			input_size = input_size[0]

		if self.dims == 3:
			DownBlock = ConvDownBlock3D
			UpBlock   = ConvUpBlock3D
		else:
			DownBlock = ConvDownBlock2D
			UpBlock   = ConvUpBlock2D
		  
		#ch = [16, 64, 256, 1024] + [z_dim]
		if not isinstance(channel_depths, list):
			ch = [16, 64, 128, 256] + [channel_depths]
			self.z_dim = channel_depths
		else:
			ch = channel_depths
			self.z_dim = ch[-1]
		
		dropout_list = dropout if isinstance(dropout, list) else [dropout for i in range(len(channel_depths))]
		insta_norm_list = insta_norm if isinstance(insta_norm, list) else [insta_norm for i in range(len(channel_depths))]
		
		latent_kernel_size = input_size//(2**(len(channel_depths)-1))

		const_params = {'stride': 2, 'scale_factor': 1, 'kernel_size': kernel_size, 'activation': activation, 'padding': 1}
		self.enc = nn.Sequential()
		for l_idx in range(len(channel_depths)):
			const_params['dropout']=dropout_list[l_idx]
			const_params['insta_norm']=insta_norm_list[l_idx]
			if l_idx == 0:
				self.enc.append(DownBlock(in_size=n_channels, out_size=ch[0], **const_params))
			elif l_idx != len(channel_depths)-1:
				self.enc.append(DownBlock(in_size=ch[l_idx-1],  out_size=ch[l_idx], **const_params))
			else: #last layer
				latent_params = {'stride': 1, 'scale_factor': 1, 'kernel_size': latent_kernel_size, 'bias':False,  'activation': activation, 'padding': 0}
				self.enc.append(DownBlock(in_size=ch[-2],     out_size=ch[-1], **latent_params))

			 
		latent_params['padding']=1 ####
		const_params['kernel_size']=2
		const_params['stride']=2
		const_params['insta_norm']=None
		const_params['dropout']=None
		self.dec = nn.Sequential() 
		for l_idx in reversed(range(len(channel_depths))):
			#const_params['dropout']=dropout_list[l_idx]
			#const_params['insta_norm']=insta_norm_list[l_idx]dec_c_mult
			if l_idx == len(channel_depths)-1: #inner-most layer
				self.dec.append(UpBlock(in_size=ch[-1]//2,  out_size=ch[-2], **latent_params))
			elif l_idx != 0:
				self.dec.append(UpBlock(in_size=ch[l_idx], out_size=ch[l_idx-1], **const_params))
			else: #last layer
				const_params['activation']=last_activation
				self.dec.append(UpBlock(in_size=ch[0],     out_size=n_classes, **const_params))
		

	def forward(self, inpt, sample=True, no_dist=True):        
		''' debug trick
		for layer in self.enc:
			inpt = layer(inpt)
		y1 = inpt
		'''
		y1 = self.enc(inpt)
		mu, log_var = torch.chunk(y1, 2, dim=1)
		std = torch.exp(log_var)
		
		z_dist = dist.Normal(mu, std)
		if sample:
			z_sample = z_dist.rsample()
		else:
			z_sample = mu
			
		#x_rec = self.dec(z_sample)
		#''' debug trick
		for layer in self.dec:
			z_sample = layer(z_sample)
		x_rec = z_sample
		#'''

		if no_dist:
			return x_rec
		else:
			return x_rec, z_dist


	def training_step(self, batch, batch_nb):
		x = batch["img"]["data"]
		shape = list(x.shape)
		try:
			index = shape[2:].index(1)
			shape.pop(2+index)
		except ValueError:
			pass
		x = x.view(*shape)
		
		y_vae_hat, z_dist = self.forward(x, no_dist=False)
		losses = self.compute_loss(x,y_vae_hat, z_dist=z_dist)
		
		real_batch_size = x.shape[0]*torch.distributed.get_world_size() if torch.distributed.group.WORLD is not None else x.shape[0]
		loss = losses.pop('loss')
		self.log('Train/'+self.loss['name'], loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		self.log('Train/loss',               loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=real_batch_size)
		for name, value in losses.items():
			self.log('Train/'+name,  value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		if self.log_wandb_images and batch_nb == 0 and self.current_epoch!=0:
			wutils.log_wandb_image(self.loggers[0], 'Train-sample/input', x)
			wutils.log_wandb_image(self.loggers[0], 'Train-sample/vae-output', y_vae_hat)
		return loss

	def validation_step(self, batch, batch_nb):
		x = batch["img"]["data"] 
		shape = list(x.shape)
		try:
			index = shape[2:].index(1)
			shape.pop(2+index)
		except ValueError:
			pass
		x = x.view(*shape)
		
		# VAE feedback
		y_vae_hat, z_dist = self.forward(x, no_dist=False)
		losses = self.compute_loss(x,y_vae_hat, z_dist=z_dist)
		loss = losses.pop('loss')
		real_batch_size = x.shape[0]*torch.distributed.get_world_size(torch.distributed.group.WORLD) if torch.distributed.group.WORLD is not None else x.shape[0]
		self.log('Valid/'+self.loss['name'],  loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		self.log("Valid/loss",        loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=real_batch_size)
		for name, value in losses.items():
			self.log('Valid/val-'+name,  value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		if self.log_wandb_images and batch_nb == 0 and self.current_epoch!=0:
			wutils.log_wandb_image(self.loggers[0], 'Valid-sample/input',  x)
			wutils.log_wandb_image(self.loggers[0], 'Valid-sample/output', y_vae_hat)
		evaluated_metrics = self.compute_additional_metrics(x,y_vae_hat, 'validation')        
		for name in evaluated_metrics:
			self.log('Valid/'+name, evaluated_metrics[name], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		return loss
	
	def validation_end(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		tensorboard_logs = {'val_loss': avg_loss}
		return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

	def test_step(self, batch, batch_nb):
		x = batch["img"]["data"]
		y = batch["label"]["data"]
		origin_shape = list(x.shape)
		shape = list(x.shape)
		try:
			index = shape[2:].index(1)
			shape.pop(2+index)
		except ValueError:
			pass
		x = x.view(*shape)
		y = y.view(*shape)
		y_hat = self.forward(x, sample=False)
		
		# If patch-based (grid), then put together the batched samples
		if self.data_module.must_aggregate_patches():
			if not hasattr(self, 'grid_sampler'): # FIXME : Careful, grid_sampler varies for each subject!
				self.set_test_grid_sampler(self.data_module.grid_sampler)  
			x     = x.view(*origin_shape)
			y     = y.view(*origin_shape)
			y_hat = y_hat.view(*origin_shape)
			locations = batch[tio.LOCATION]
			layered_batch = torch.cat((x, y, y_hat), dim=1)
			self.output_aggregator.add_batch(layered_batch, locations)
			return 0
		#else calculate directly the losses
		else:
			processed = self.hook_ex_external_code('test', 'post_processing', x=x, y_hat=y_hat)
			y_hat = processed['y_hat']
			y_hat = torch.abs(y_hat-x)
			batch_metrics = []
			batch_metrics_to_write = dict()
			for sample_idx in range(x.shape[0]):
				batch_metrics.append(self.compute_additional_metrics(y[sample_idx,...],y_hat[sample_idx,...], 'test', x=x[sample_idx,...]))
				for name, value in batch_metrics[-1].items():
					batch_metrics_to_write['Test-step/'+name] = value
				self.loggers[0].log_metrics(batch_metrics_to_write)
			return batch_metrics

	def predict_step(self, batch, batch_nb):
		x = batch["img"]["data"]
		origin_shape = list(x.shape)
		shape = list(x.shape)
		try:
			index = shape[2:].index(1)
			shape.pop(2+index)
		except ValueError:
			pass
		x = x.view(*shape)
		y_hat = self.forward(x,sample=False)

		# If patch-based (grid), then put together the batched samples
		if self.data_module.must_aggregate_patches():
			if not hasattr(self, 'grid_sampler'): # FIXME : Careful, grid_sampler varies for each subject!
				self.set_test_grid_sampler(self.data_module.grid_sampler)
			x     = x.view(*origin_shape)
			y_hat = y_hat.view(*origin_shape)
			locations = batch[tio.LOCATION]
			layered_batch = torch.cat((x, y_hat), dim=1)
			self.output_aggregator.add_batch(layered_batch, locations)
		else:
			names = batch["name"]
			y_hat = self.hook_ex_external_code('test', 'post_processing', input=x, y_hat=y_hat)
			
			x = torch.squeeze(x,dim=1)
			y_hat = torch.squeeze(y_hat,dim=1)

			reco_tmp = abs(x - y_hat)
			#reco_tmp = self.hook_ex_external_code('test', 'post_processing', [reco_tmp, y_hat, x])
			reco_tmp = reco_tmp.cpu().detach().numpy()
			infer_tmp = y_hat.cpu().detach().numpy()
			
			output_dir=self.data_module.test_and_predict['dataset'].get('results')
			pathlib.Path(output_dir+'/infer').mkdir(parents=True, exist_ok=True) 
			pathlib.Path(output_dir+'/rec_error').mkdir(parents=True, exist_ok=True) 
			
			for subj in range(x.shape[0]):
				img = nib.Nifti1Image(infer_tmp[subj,...], np.eye(4))
				img.header.get_xyzt_units()
				img.to_filename(os.path.join(output_dir, 'infer', names[subj] + '.nii.gz')) 
				
				img = nib.Nifti1Image(reco_tmp[subj,...], np.eye(4))
				img.header.get_xyzt_units()
				img.to_filename(os.path.join(output_dir, 'rec_error', names[subj] + '.nii.gz'))             
				#print('Saved in '+os.path.join(output_dir, 'infer.nii.gz'))
			
		return 0

	def test_epoch_end(self, outputs):
		if self.data_module.must_aggregate_patches(): 
			output_tensor = self.output_aggregator.get_output_tensor()
			x = output_tensor[0,:,:,:]
			y = output_tensor[1,:,:,:]
			y_hat = output_tensor[2,:,:,:]
			rec_error = torch.abs(x - y_hat)
			del self.output_aggregator, output_tensor
			gc.collect()
			#device = 'cuda:'+str(ut.find_free_gpu()[0])
			y_hat     = y_hat.to(device=self.device)
			rec_error = rec_error.to(device=self.device)
			processing_args = self.data_module.test.get('post_processing')
			processed = common.hook_ex_external_code(processing_args=processing_args, y_hat=y_hat, rec_error=rec_error)
			rec_error = processed.get('rec_error', rec_error)
			#rec_error = rec_error.to(device=self.device)
			#y = y.to(device=self.device)
			gc.collect()
			metrics = self.compute_additional_metrics(y,rec_error, 'test', x=x)
			common.save_results_on_disk(self.data_module.test_and_predict['dataset'].get('results'), metrics)
			for name in metrics:
				self.log('Test/'+name, metrics[name]) 
			'''
			y2 = y.cpu().detach().numpy()
			y2 = y2.flatten()
			y_hat2 = y_hat.cpu().detach().numpy()
			y_hat2 = y_hat2.flatten()
			precision, recall, auprc_value, pr_thresholds, tpr, fpr, auc_value, roc_thresholds = find_curve_metrics(y2,y_hat2)
			self.save_values(self.name, precision, recall, auprc_value, pr_thresholds, tpr, fpr, auc_value, roc_thresholds)
			'''
			# wandb table
			columns = [name for name in metrics]
			data = [] 
			for name in metrics:
				data.append([metrics[name] for name in metrics])
			self.loggers[0].log_table(key='Test/scores_table', columns=columns, data=data)
		else:
			grouped_values = dict()
			n_samples = 0
			for batch_metrics in outputs:
				for metrics in batch_metrics:
					n_samples+=1
					for name in metrics: 
						grouped_values['Test/'+name] = grouped_values.get('Test/'+name,0) + metrics[name]
			for key,value in grouped_values.items():
				grouped_values[key] = value/n_samples
			self.loggers[0].log_metrics(grouped_values,step=0)
			columns = [name for name in outputs[0][0].keys()]
			data = [] 
			for batch_metrics in outputs:
				for metrics in batch_metrics:
					data.append([metrics[name] for name in metrics])
			self.loggers[0].log_table(key='Test/scores_table', columns=columns, data=data)
		
	def on_predict_epoch_end(self, outputs):
		if self.data_module.must_aggregate_patches():
			print('Saving..')
			output_tensor = self.output_aggregator.get_output_tensor()
			x = output_tensor[0,:,:,:]
			y_hat = output_tensor[1,:,:,:]
			rec_error = torch.abs(x-y_hat)
			processing_args = self.data_module.predict.get('post_processing')
			processed = common.hook_ex_external_code(processing_args=processing_args,  y_hat=y_hat, rec_error=rec_error)
			rec_error = processed['rec_error']
			tmp = rec_error.cpu().detach().numpy()
			tmp = np.squeeze(tmp)
			tmp = np.transpose(tmp,axes=(1,0,2))
			output_dir = self.data_module.test_and_predict['dataset'].get('results')
			if output_dir is not None:
				pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True) 
				for slice_number in range(tmp.shape[2]):
					im = Image.fromarray(tmp[:,:,slice_number])
					im.save(os.path.join(output_dir, 'infer_'+str(slice_number)+'.tif'))
			else:
				raise Exception('Output path for the prediction has not been set')
		else:
			pass

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
		self.output_aggregator = tio.inference.GridAggregator(self.grid_sampler, overlap_mode='average') #crop/average

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
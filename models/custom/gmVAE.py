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
			self.up = nn.ConvTranspose3d(in_size, in_size, kernel_size=scale_factor,
										 stride=scale_factor)
		elif up_mode == 'upsample':
			#self.up = nn.Sequential(nn.Upsample(mode='trilinear', scale_factor=scale_factor, align_corners=False),
			self.up = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=scale_factor),
									nn.Conv3d(in_size, in_size, kernel_size=1))

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

	def forward(self, x):
		up = self.up(x)
		out = self.conv_block(up)
		return out

def conv_block(in_channels, out_channels, kernel_size, padding, stride, activation='relu', bias=True):
	block = nn.Sequential(
		nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
	)
	if activation.casefold() == 'prelu':
		nn.init.kaiming_normal_(block[-1].weight)
		block.append(nn.PReLU())
	elif activation.casefold() == 'leakyrelu':
		nn.init.kaiming_normal_(block[-1].weight)
		block.append(nn.LeakyReLU(0.01))
	elif activation.casefold() == 'sigmoid':
		nn.init.xavier_uniform_(block[-1].weight) # works well for sigmoid
		block.append(nn.Sigmoid())
	return block

def transp_conv_block(in_channels, out_channels, kernel_size, padding, stride, output_padding = 0, activation=nn.ReLU()):
	block = nn.Sequential(
		nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding),
	)
	if activation.casefold() == 'prelu':
		nn.init.kaiming_normal_(block[-1].weight)
		block.append(nn.PReLU())
	elif activation.casefold() == 'leakyrelu':
		nn.init.kaiming_normal_(block[-1].weight)
		block.append(nn.LeakyReLU(0.01))
	elif activation.casefold() == 'sigmoid':
		nn.init.xavier_uniform_(block[-1].weight) # works well for sigmoid
		block.append(nn.Sigmoid())
	return block


class gmVAE(pl.LightningModule):
	def __init__(
		self, name, checkpoint_path=None, log_path=None, n_channels=1, n_classes=1, depth=3, wf=6, padding=True, patch_size=64, z_dim=1024, flat_latent=True, scale_factor=2,
		batch_norm=False, up_mode='upconv', activation='LeakyReLu', loss={'name': 'cross_entropy'}, channel_depths=None, dim_c=60, dim_z=500, dim_w=1,
		evaluate_metrics={}, dropout=None, last_activation='', optimizer_parameters=None, optimizer = 'RMSprop', input_size=256,  **kwargs
	):
		super(gmVAE, self).__init__()
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
		self.scale_factor=scale_factor
		self.optimizer_param = optimizer_parameters
		self.loss = loss
		self.eval_metrics = evaluate_metrics
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
		'''
		if activation == 'relu':
			activation = nn.ReLU()
		elif activation == 'leakyrelu':
			activation = nn.LeakyReLU()
		
		if last_activation == 'relu':
			last_activation = nn.ReLU()
		elif last_activation == 'leakyrelu':
			last_activation = nn.LeakyReLU()
		elif last_activation == 'sigmoid':
			last_activation = nn.Sigmoid()'''
		
		self.dim_c = dim_c #9 # n clusters
		self.dim_z = dim_z #1
		self.dim_w = dim_w #1

	
		'''self.conv_block1 = conv_block(in_channels = 1,  out_channels = 64, kernel_size = 3, padding = 1, stride = 2, activation=activation)
		self.conv_block2 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1, activation=activation)
		self.conv_block3 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1, activation=activation)
		self.conv_block4 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 2, activation=activation)
		self.conv_block5 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1, activation=activation)
		self.conv_block6 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1, activation=activation)'''


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
		
		self.down_path = nn.ModuleList()
		for i,end_channels in enumerate(channel_depths):
			if i != 0:
				self.down_path.append(UNetConvBlock(channel_depths[i-1], end_channels,
												padding=padding, batch_norm=batch_norms[i], activation=activation, 
												dropout=dropouts[i], stride=1))
			else:
				self.down_path.append(UNetConvBlock(n_channels, end_channels,
												padding=padding, batch_norm=batch_norms[i], activation=activation, 
												dropout=dropouts[i], stride=1))

		c = channel_depths[-1] # bottleneck channel

		self.w_mu_layer        = nn.Conv3d(c, self.dim_w, kernel_size = 1, padding = 0, stride = 1, bias=False) #
		self.w_log_sigma_layer = nn.Conv3d(c, self.dim_w, kernel_size = 1, padding = 0, stride = 1, bias=False) #
		
		self.z_mu_layer        = nn.Conv3d(c, self.dim_z, kernel_size = 1, padding = 0, stride = 1, bias=False)
		self.z_log_sigma_layer = nn.Conv3d(c, self.dim_z, kernel_size = 1, padding = 0, stride = 1, bias=False)
		
		self.conv_block7 = conv_block(in_channels = self.dim_w, out_channels = c, kernel_size = 1, padding = 0, stride = 1, activation=activation)
		self.z_wc_mu_layer        = nn.Conv3d(c, self.dim_z * self.dim_c, kernel_size = 1, padding = 0, stride = 1, bias=False) #
		self.z_wc_log_sigma_layer = nn.Conv3d(c, self.dim_z * self.dim_c, kernel_size = 1, padding = 0, stride = 1, bias=False) #
		
		self.conv_block8 = conv_block(in_channels = self.dim_z, out_channels = c, kernel_size = 1, padding = 0, stride = 1, activation=activation)
		self.up_path = nn.ModuleList()
		for i,end_channels in enumerate(reversed(channel_depths[:-1])):
			self.up_path.append(UNetUpBlock(channel_depths[-1-i], end_channels, up_mode=up_mode, 
											padding=padding, batch_norm=batch_norms[i], activation=activation, 
											dropout=dropouts[i],scale_factor=scale_factor))

		self.xz_mu_layer = conv_block(in_channels = channel_depths[0], out_channels=n_classes, kernel_size = 3, padding = 1, stride = 1, activation=last_activation)


	def forward(self, image, sample=True): #FIXME: sample
		outputs = {}
		
		x=image
		for i, down in enumerate(self.down_path):
			x = down(x)
			if i != len(self.down_path)-1:
				x = F.avg_pool3d(x, self.scale_factor)        
		
		outputs['z_mu'] = z_mu = self.z_mu_layer(x)
		outputs['z_log_sigma'] = z_log_sigma = self.z_log_sigma_layer(x)
		# reparametrization
		if sample:
			rand_z = torch.randn(z_log_sigma.shape, device=self.device) * torch.exp(0.5 * z_log_sigma)
		else:
			rand_z =z_mu
		outputs['z_sampled'] = z_sampled = z_mu + rand_z

		outputs['w_mu']        = w_mu = self.w_mu_layer(x)
		outputs['w_log_sigma'] = w_log_sigma = self.w_log_sigma_layer(x)
		rand_w = torch.randn(w_log_sigma.shape, device=self.device) * torch.exp(0.5 * w_log_sigma)
		outputs['w_sampled'] = w_sampled = w_mu + rand_w

		# posterior p(z|w,c)
		x7 = self.conv_block7(w_sampled)
		z_wc_mu = self.z_wc_mu_layer(x7)
		z_wc_log_sigma = self.z_wc_log_sigma_layer(x7)
		bias = torch.full_like(z_wc_log_sigma, 0.1)
		z_wc_log_sigma_inv = z_wc_log_sigma + bias
		outputs['z_wc_mus'] = z_wc_mus = z_wc_mu.view(-1, self.dim_c, self.dim_z, *z_wc_mu.shape[2:])
		outputs['z_wc_log_sigma_invs'] = z_wc_log_sigma_invs = z_wc_log_sigma_inv.view(-1, self.dim_c, self.dim_z, *z_wc_log_sigma_inv.shape[2:])
		rand_z_wc = torch.randn(z_wc_log_sigma_invs.shape, device=self.device) * torch.exp(z_wc_log_sigma_invs)
		outputs['z_wc_sampled'] = z_wc_sampled = z_wc_mus + rand_z_wc
		
		# decoder p(x|z)
		y = self.conv_block8(z_sampled)
		for i, up in enumerate(self.up_path):
			y = up(y) 
		outputs['xz_mu'] = xz_mu = self.xz_mu_layer(y)

		# prior p(c)
		z_sample_shape = [1 for x in z_wc_mus.shape]
		z_sample_shape[1] = self.dim_c
		z_sample = z_sampled.unsqueeze(dim = 1).repeat(z_sample_shape)
		loglh = -0.5 * (((z_sample - z_wc_mus) ** 2) * torch.exp(z_wc_log_sigma_invs)) - z_wc_log_sigma_invs + torch.log(torch.tensor(np.pi))
		loglh_sum = torch.sum(loglh, dim = 2)
		outputs['pc_logit'] = pc_logit = loglh_sum
		outputs['pc'] = pc = nn.Softmax(dim = 1)(loglh_sum)
		
		return outputs
	
	
	def training_step(self, batch, batch_nb):
		x = batch["img"]["data"]
		shape = list(x.shape)
		try:
			index = shape[2:].index(1)
			shape.pop(2+index)
		except ValueError:
			pass
		x = x.view(*shape)
		output = self.forward(x)
		y_hat = output.pop('xz_mu')
		losses = self.compute_loss(x,y_hat, **output)
		real_batch_size = x.shape[0]*torch.distributed.get_world_size() if torch.distributed.group.WORLD is not None else x.shape[0]
		loss = losses.pop('loss')
		self.log('Train/'+self.loss['name'], loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		self.log('Train/loss',               loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=real_batch_size)
		for name, value in losses.items():
			self.log('Train/'+name,  value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		if self.log_wandb_images and batch_nb == 0 and self.current_epoch!=0:
			wutils.log_wandb_image(self.loggers[0], 'Train-sample/input', x)
			wutils.log_wandb_image(self.loggers[0], 'Train-sample/output', y_hat)
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
  
		output = self.forward(x, sample=False)
		y_hat = output.pop('xz_mu')
		losses = self.compute_loss(x,y_hat, **output)
		loss = losses.pop('loss')
		real_batch_size = x.shape[0]*torch.distributed.get_world_size(torch.distributed.group.WORLD) if torch.distributed.group.WORLD is not None else x.shape[0]
		self.log('Valid/'+self.loss['name'],  loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		self.log("Valid/loss",        loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=real_batch_size)
		for name, value in losses.items():
			self.log('Valid/val-'+name,  value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		if self.log_wandb_images and batch_nb == 0 and self.current_epoch!=0:
			wutils.log_wandb_image(self.loggers[0], 'Valid-sample/input',  x)
			wutils.log_wandb_image(self.loggers[0], 'Valid-sample/output', y_hat)
		evaluated_metrics = self.compute_additional_metrics(x,y_hat, 'validation')   
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
		output = self.forward(x,sample=False)
		y_hat = output.pop('xz_mu')
  
		if not torch.isfinite(y_hat).all():
			print("x is not finite")
   
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
			y_hat = abs(y_hat-x)
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
		output = self.forward(x)
		y_hat = output['xz_mu']

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
		if self.optimizer.casefold() == 'adamw':
			return torch.optim.AdamW(self.parameters(), **self.optimizer_param)
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
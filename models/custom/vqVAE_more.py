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
# This file incorporates work covered originally by the Sonnet package
# (https://github.com/deepmind/sonnet), which is licensed as follows:
#
# 	Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# 	Licensed under the Apache License, Version 2.0 (the "License");
# 	you may not use this file except in compliance with the License.
# 	You may obtain a copy of the License at
#
#    	http://www.apache.org/licenses/LICENSE-2.0
#
# 	Unless required by applicable law or agreed to in writing, software
# 	distributed under the License is distributed on an "AS IS" BASIS,
# 	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# 	See the License for the specific language governing permissions and
# 	limitations under the License.
# ============================================================================
#
# The original content has been modified:
#
# 7 June 2023 - imec-Vision Lab, University of Antwerp: Added support for 3D data, add GPL3 license

import os, pathlib, gc, time, numpy as np, nibabel as nib, matplotlib.pyplot as plt, PIL.Image as Image
from functools import partial

import torch, torch.nn as nn, torch.nn.functional as F, torch.distributions as dist
import pytorch_lightning as pl, torchio as tio, wandb
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from utils.stats import *
import utils.wandb as wutils
from utils.extern import *
from nn.models.helpers import common_operations as common
from nn.models.helpers.aes import Encoder_more, Decoder_more

# This vqVAE uses more encoder and decoder stages in order to reduce the image size more

class VectorQuantizerEMA(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
		super(VectorQuantizerEMA, self).__init__()
		
		self._embedding_dim = embedding_dim
		self._num_embeddings = num_embeddings
		
		self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
		self._embedding.weight.data.normal_()
		self._commitment_cost = commitment_cost
		
		self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
		self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
		self._ema_w.data.normal_()
		
		self._decay = decay
		self._epsilon = epsilon

	def forward(self, inputs):
		# convert inputs from BCHW -> BHWC
		#inputs = inputs.permute(0, 2, 3, 1).contiguous()
		inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
		input_shape = inputs.shape
		
		# Flatten input
		flat_input = inputs.view(-1, self._embedding_dim)
		
		# Calculate distances
		distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
					+ torch.sum(self._embedding.weight**2, dim=1)
					- 2 * torch.matmul(flat_input, self._embedding.weight.t()))
			
		# Encoding
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
		encodings.scatter_(1, encoding_indices, 1)
		
		# Quantize and unflatten
		quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
		
		# Use EMA to update the embedding vectors
		if self.training:
			self._ema_cluster_size = self._ema_cluster_size * self._decay + \
									 (1 - self._decay) * torch.sum(encodings, 0)
			
			# Laplace smoothing of the cluster size
			n = torch.sum(self._ema_cluster_size.data)
			self._ema_cluster_size = (
				(self._ema_cluster_size + self._epsilon)
				/ (n + self._num_embeddings * self._epsilon) * n)
			
			dw = torch.matmul(encodings.t(), flat_input)
			self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
			
			self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
		
		# Loss
		e_latent_loss = F.mse_loss(quantized.detach(), inputs)
		loss = self._commitment_cost * e_latent_loss
		
		# Straight Through Estimator
		quantized = inputs + (quantized - inputs).detach()
		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
		
		# convert quantized from BHWC -> BCHW
		#return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
		return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings

class vqVAE_more(pl.LightningModule):

	def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
				num_embeddings, embedding_dim, commitment_cost, name, decay=0,
				checkpoint_path=None, log_path=None, n_channels=1, n_classes=1, 
				input_size=256, patch_size=64, channel_depths=1024, loss={'name': 'cross_entropy'}, 
				evaluate_metrics={}, dropout=None, last_activation='sigmoid', optimizer_parameters=None, optimizer = 'RMSprop', **kwargs):
		super(vqVAE_more, self).__init__()
		self.save_hyperparameters()
		self.name = name
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.checkpoint_path = checkpoint_path
		self.log_path = log_path
		self.last_activation = last_activation
		self.input_size = input_size
		self.optimizer = optimizer
		self.optimizer_param = optimizer_parameters
		self.loss = loss
		self.eval_metrics = evaluate_metrics
		self.log_wandb_images = kwargs.get('log_wandb_images', False)

		self._encoder = Encoder_more(n_channels, num_hiddens,
								num_residual_layers, 
								num_residual_hiddens)
		self._pre_vq_conv = nn.Conv3d(in_channels=num_hiddens, 
									  out_channels=embedding_dim,
									  kernel_size=1, 
									  stride=1)
		self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
											commitment_cost, decay)
		self._decoder = Decoder_more(embedding_dim,
								num_hiddens, 
								num_residual_layers, 
								num_residual_hiddens,
        						n_classes=n_classes)
  
	def forward(self, x):
		z = self._encoder(x)
		z = self._pre_vq_conv(z)
		loss, quantized, perplexity, _ = self._vq_vae(z)
		x_recon = self._decoder(quantized)
		return loss, x_recon, perplexity


	def training_step(self, batch, batch_nb):
		x = batch["img"]["data"]
		shape = list(x.shape)
		try:
			index = shape[2:].index(1)
			shape.pop(2+index)
		except ValueError:
			pass
		x = x.view(*shape)
		vq_loss, y_hat, perplexity = self.forward(x)  
		losses = self.compute_loss(x,y_hat, vq_loss=vq_loss, perplexity=perplexity)
		real_batch_size = x.shape[0]*torch.distributed.get_world_size() if torch.distributed.group.WORLD is not None else x.shape[0]
		loss = losses.pop('loss')
		self.log('Train/'+self.loss['name'], loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		self.log('Train/loss',               loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=real_batch_size)
		for name, value in losses.items():
			self.log('Train/'+name,  value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,batch_size=real_batch_size)
		if self.log_wandb_images and batch_nb == 0 and self.current_epoch!=0:
			wutils.log_wandb_image(self.loggers[0], 'Train-sample/input', x)
			wutils.log_wandb_image(self.loggers[0], 'Train-sample/vae-output', y_hat)
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
		vq_loss, y_hat, perplexity = self.forward(x)  
		losses = self.compute_loss(x,y_hat, vq_loss=vq_loss, perplexity=perplexity)
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
		_, y_hat, _ = self.forward(x)
		
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
			#outputs = batch_metrics

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
		_, y_hat, _ = self.forward(x)

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

	def on_test_epoch_end(self):
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
			#outputs should be a concatenation of all the outputs of the test_step. loss due to pytorch lightning update
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
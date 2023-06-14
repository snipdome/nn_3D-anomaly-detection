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


from utils import stats
from utils import extern
import torch

import matplotlib.pyplot as plt

def hook_ex_external_code(processing_args, **input):
	'''
	This function call internal and external code for processing of the input, which can be a series of torch/numpy arrays.
	
	'''
	if processing_args is not None:
		
		# little hack to allow several processing
		if not isinstance(processing_args, list):
			processing_args = [processing_args]
		
			
		for tool_specific_processing_args in processing_args:
			inputs_to_forward = tool_specific_processing_args.pop('forward_arrays', 'input')
			if not isinstance(inputs_to_forward, list):
				inputs_to_forward = [inputs_to_forward]
			
			# Since the external code may not support gpu, everything goes in numpy by default
			gpu_list={} # store the device of each volume before the processing
			for input_to_forward in inputs_to_forward:
				gpu_list[input_to_forward] = input[input_to_forward].device
				input[input_to_forward]    = input[input_to_forward].cpu().detach().numpy()
			
			forward_devices = tool_specific_processing_args.pop('forward_devices', True)
			if forward_devices == True:
				forward_devices = gpu_list
			elif isinstance(forward_devices, list):
				forward_devices = {input_to_forward: forward_devices[i] for i,input_to_forward in enumerate(inputs_to_forward)}
			else:
				forward_devices = {}
			print('Executing external code: {}.{}'.format(tool_specific_processing_args['module'], tool_specific_processing_args['fun']))
			res = extern.call_external_code( 
				gpu_list=forward_devices, # In case the end-program can use gpus
				**{k: input.get(k, None) for k in inputs_to_forward}, 
				**tool_specific_processing_args)

			if not isinstance(res, dict):
				output = {inputs_to_forward[0]:res} # By default, take the first forwarded input as name of the output
			else:
				output = {k: res.get(k, input.get(k, None)) for k in inputs_to_forward} #otherwise, the output is the same as the input if there is no processing (res)
				
			# Bring the code back on device
			for key,value in output.items():
				if not torch.is_tensor(value):
					output[key] = torch.tensor(value, device=gpu_list[key])
				else:
					value = value.detach().cpu() if value.device != 'cpu' else value
					output[key] = value.to(device=gpu_list[key])
			input.update(output)
			
		output = input
		return output
	else:
		return input

def save_results_on_disk(path, results):
	if results.get('pore_segmentation_curves') is not None:
		import pathlib, pickle, os
		print('Saving output in '+path)
		pathlib.Path(path).parent.absolute().mkdir(parents=True, exist_ok=True) 
		with open( path+'.pickle' , 'wb') as f:
			pickle.dump(results, f)
		results['pore_segmentation_curves'] = results['pore_segmentation_curves'].get('average_precision')


def compute_metric(y, y_hat, metric_to_eval, **additional_inputs):
	parameters = metric_to_eval.get('parameters',{})
	if parameters is None:
		parameters = {}
	if metric_to_eval['name'] in ['dice_score', 'dice']:
		loss = stats.dice(y, y_hat, **parameters)
	elif metric_to_eval['name'] == 'skipAE_loss':
		loss = stats.skipAE_loss(y,y_hat, **parameters)
	elif metric_to_eval['name'] == 'vqVAE_loss':
		loss = stats.vqVAE_loss(y, y_hat, **additional_inputs, **parameters)
	elif metric_to_eval['name'] == 'ceVAE_loss':
		loss = stats.ceVAE_loss(y, y_hat, **additional_inputs, **parameters)
	elif metric_to_eval['name'] == 'hd_ceVAE_loss':
		loss = stats.hd_ceVAE_loss(y, y_hat, **additional_inputs, **parameters)
	elif metric_to_eval['name'] == 'VAE_loss':
		loss = stats.VAE_loss(y, y_hat, **additional_inputs, **parameters)
	elif metric_to_eval['name'] == 'gmVAE_loss':
		loss = stats.gmVAE_loss(y, y_hat, **additional_inputs, **parameters)
	elif metric_to_eval['name'] == 'torch_dice':
		loss = stats.torch_dice(y, y_hat, **parameters)       
	elif metric_to_eval['name'] == 'dice_loss':
		loss = stats.dice_loss(y, y_hat, **parameters)    
	elif metric_to_eval['name'] == 'torch_dice_loss':
		loss = stats.torch_dice_loss(y, y_hat, **parameters)        
	elif metric_to_eval['name'] == 'cross_entropy':
		loss = stats.F.cross_entropy(y,y_hat)
	elif metric_to_eval['name'] == 'mse_loss':
		loss = stats.F.mse_loss(y,y_hat)
	elif metric_to_eval['name'] == 'tversky_loss':
		loss = stats.tversky_loss(y, y_hat, **parameters)
	elif metric_to_eval['name'] == 'focal_tversky_loss':
		loss = stats.focal_tversky_loss(y, y_hat, **parameters)
	elif metric_to_eval['name'] == 'torch_focal_tversky_loss':
		loss = stats.torch_focal_tversky_loss(y, y_hat, **parameters)
	elif metric_to_eval['name'] == 'torch_auc_roc':
		loss = stats.torch_auc_roc(y, y_hat, **parameters)
	elif metric_to_eval['name'] == 'torch_average_precision':
		loss = stats.torch_average_precision(y, y_hat, **parameters)
	elif metric_to_eval['name'] == 'Hausdorff_distance':
		loss = stats.Hausdorff_distance(y, y_hat, **parameters)
	elif metric_to_eval['name'] == 'average_Hausdorff_distance':
		loss = stats.average_Hausdorff_distance(y, y_hat, **parameters)
	elif metric_to_eval['name'] == 'pore_segmentation_curves':
		loss = stats.pore_segmentation_curves(y, y_hat, **additional_inputs, **parameters)
	else:
		raise Exception('loss \"'+str(metric_to_eval['name'])+'\" not yet implemented')
	return loss
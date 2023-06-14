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


from cmath import nan
import torch, torch.nn.functional as F, torch.distributions as dist, numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from torchmetrics import functional as TMF
import torchmetrics#, evaluate_segmentation

from sklearn.utils.class_weight import compute_sample_weight




def plot_recall_precision(recall, precision, auprc_values, labels):
	_, ax = plt.subplots(figsize=(7, 8))
	f_scores = np.linspace(0.2, 0.8, num=4)
	lines = []
	for f_score in f_scores:
		x = np.linspace(0.01, 1)
		y = f_score * x / (2 * x - f_score)
		(l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
		plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

	#target = np.random.randint(2, size=1000)*np.random.randint(2, size=1000)*np.random.randint(2, size=1000)
	#pred   = 0.5*target+np.random.uniform(low=-0.5, high=0.5, size=1000)
	#precision, recall, auprc_value, pr_thresholds, tpr, fpr, auc_value, roc_thresholds = find_metrics(target, pred)
	for i in range(len(recall)):
		ax.plot(recall[i], precision[i],label=labels[i]+' - auc: '+str(auprc_values[i]))

	# set the legend and the axes
	ax.set_xlim([0.0, 1.05])
	ax.set_ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	#ax.legend(handles=handles, labels=labels, loc="best")
	ax.legend(loc='best')
	ax.set_title("Precision-Recall curve")

	plt.show()

def plot_roc(tpr, fpr, auc_values, labels):
	
	_, ax = plt.subplots(figsize=(7, 7))
	
	for i in range(len(tpr)):
		ax.plot(fpr[i], tpr[i],label=labels[i]+' - auc: '+str(auc_values[i]))
	
	plt.plot([0, 1], [0, 1],'k--')
	ax.set_xlim([0.0, 1.05])
	ax.set_ylim([0.0, 1.05])
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	ax.legend(loc='best')
	ax.set_title("ROC curve")
	plt.show()

def find_curve_metrics(target, pred, ds=50):
	target = target[::ds].astype(int)
	pred = pred[::ds]
	#target = target.astype(int)
	
	#n_true = np.sum(target)
	#total = np.prod(target.shape)
	#print('n of voxels {} and n of true is {}'.format(total,n_true))
	sample_weight = compute_sample_weight(class_weight='balanced', y=target)
	#precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred, sample_weight=sample_weight)
	precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred)
	auprc_value = metrics.auc(recall, precision)
	print(auprc_value)
	fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred, pos_label=1)
	auc_value  = metrics.auc(fpr, tpr)
	print(auc_value)
	#metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None)
	return precision, recall, auprc_value, pr_thresholds, tpr, fpr, auc_value, roc_thresholds

def Hausdorff_distance(target, pred, quantile=1, fuzzy=False, threshold=-1, millimeter=True):
	if torch.is_tensor(target):
		a1 = target.cpu().detach().numpy()
		a2 = pred.cpu().detach().numpy()
	else:
		a1 = target
		a2 = pred
	nx, ny, nz = target.shape
	a1 = np.ascontiguousarray(a1.flatten())
	a2 = np.ascontiguousarray(a2.flatten())
	
	if threshold == 'otsu':
		a2 = apply_otsu(a2)
		threshold = -1
		
	a1 = 255*a1.astype(np.float32)
	a2 = 255*a2.astype(np.float32)
	
	hd_metric = evaluate_segmentation.PyHausdorffDistanceMetric(a1,a2,nx,ny,nz, fuzzy, threshold, millimeter)
	res = hd_metric.CalcHausdorffDistace(quantile)
	return res

def average_Hausdorff_distance(target, pred, prune=True, fuzzy=False, threshold=-1, millimeter=True):
	if torch.is_tensor(target):
		a1 = target.cpu().detach().numpy()
		a2 = pred.cpu().detach().numpy()
	else:
		a1 = target
		a2 = pred
	nx, ny, nz = target.shape
	a1 = np.ascontiguousarray(a1.flatten())
	a2 = np.ascontiguousarray(a2.flatten())
	
	if threshold == 'otsu':
		a2 = apply_otsu(a2)
		threshold = -1
		
	a1 = 255*a1.astype(np.float32)
	a2 = 255*a2.astype(np.float32)
	
	hd_metric = evaluate_segmentation.PyAverageDistanceMetric(a1,a2,nx,ny,nz, fuzzy, threshold, millimeter)
	res = hd_metric.CalcAverageDistace(prune)
	return res

def auc(target, pred, **kwargs):
	'''
	fpr, tpr, thresholds = metrics.roc_curve(target.cpu().detach().numpy().flatten(), pred.cpu().detach().numpy().flatten())
	auc_value = metrics.auc(fpr, tpr)
	
	plt.figure()
	lw = 2
	plt.plot(
		fpr[2],
		tpr[2],
		color="darkorange",
		lw=lw,
		label="ROC curve (area = %0.2f)" % auc_value[2],
	)
	plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Receiver operating characteristic example")
	plt.legend(loc="lower right")
	plt.show()
	'''
	debug = False
	target = target.cpu().detach().numpy()
	target = target.astype(int)
	if debug:
		plt.figure()
		target3=target
		plt.imshow(target3[:,:,300])
	target = target.flatten()
	target = target[::10]
	pred = pred.cpu().detach().numpy()
	if debug:
		plt.figure()
		pred3=pred
		plt.imshow(pred3[:,:,300])
		plt.show()
	pred = pred.flatten()
	pred = pred[::10]
	lr_precision, lr_recall, _ = metrics.precision_recall_curve(target, pred)
	auc_value = metrics.auc(lr_recall, lr_precision)
	
	plt.figure()
	lw = 2
	no_skill = len(target[target==1]) / len(target)
	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	plt.plot(lr_recall, lr_precision, marker='.', label='pores')
	plt.xlabel("Recall Rate")
	plt.ylabel("Precision Rate")
	plt.title("Recall-precision curve of ceVAE")
	plt.legend(loc="upper right")
	
	return auc_value

def torch_auc_roc(target, pred, compute_on_cpu=False, **args):    
	#print('Calculating AUC_roc')
	''' scikit
	target = target.cpu().detach().numpy()
	target = target.flatten()
	pred = pred.cpu().detach().numpy()
	pred = pred.flatten()
	try:
		auc = metrics.roc_auc_score(target, pred)
	'''
	pred2 = pred.type(torch.float32) 
	target2 = target.type(torch.int32) 
	device = pred2.device
	if compute_on_cpu == True:
		pred2 = pred2.detach().cpu()
		target2 = target2.detach().cpu()
		device = 'cpu'
	
	pred2 = pred2.view(-1,1)
	target2 = target2.view(-1,1)
	#pred2 = torch.unsqueeze(pred2, dim=1)
	#pred2 = torch.cat((abs(1-pred2),pred2),dim=1)
	try:
		#auc = TMF.auroc(preds=pred2, target=target2, **args)
		auroc = torchmetrics.AUROC(**args).to(device=pred.device)
		res = auroc(preds=pred2, target=target2)
	except ValueError as e:
		print(e)
		res = float("nan")
	except RuntimeError:
		if device != 'cpu':
			print('Error performing on GPU. retrying on CPU')
			res = torch_auc_roc(target=target2, pred=pred2, compute_on_cpu=True, **args)
	return res


def torch_average_precision(target, pred, compute_on_cpu=False, **args):    
	#print('Calculating AVG_prec')
	pred2 = pred.type(torch.float32) 
	target2 = target.type(torch.int32) 
	device = pred2.device
	if compute_on_cpu == True:
		pred2 = pred2.detach().cpu()
		target2 = target2.detach().cpu()
		device = 'cpu'
	
	pred2 = pred2.view(-1,1)
	target2 = target2.view(-1,1)
	#pred2 = torch.unsqueeze(pred2, dim=1)
	#pred2 = torch.cat((abs(1-pred2),pred2),dim=1)
	#return TMF.average_precision(preds=pred2,target=target2, **args)
	average_precision_fnc = torchmetrics.AveragePrecision(**args).to(device=pred.device)
	try:
		res = average_precision_fnc(preds=pred2, target=target2)
	except RuntimeError:
		if device != 'cpu':
			print('Error performing on GPU. retrying on CPU')
			res = torch_average_precision(target=target2, pred=pred2, compute_on_cpu=True, **args)
	return res
	

from skimage import filters

def apply_otsu(volume):
	#device = volume.device
	#volumeShape = volume.shape
	device = None
	if torch.is_tensor(volume):
		device = volume.device
		shape = volume.shape
		volume2 = volume.view(-1,1)
		volume2 = volume2.cpu().detach().numpy()
	else:
		volume2 = volume
	res = filters.threshold_otsu(volume2) #
	volume = 1.0*(volume>res)
	return volume

def threshold_selector(input, threshold):
	if threshold is None:
		output = input
	elif isinstance(threshold, (float, int)) or torch.is_tensor(threshold):
		output = 1*(input>threshold)
	elif threshold.lower() == 'otsu':
		print('With otsu thresholding..')
		output = apply_otsu(input)
	elif threshold is not None:
		raise Exception('Requested threshold has not been yet implemented.')
	return output
	

def dice(target, pred, threshold=None, compute_on_cpu=False, zero_division=1e-4, **args):
	pred2 = threshold_selector(pred,threshold)
	device = target.device
	if compute_on_cpu == True:
		pred2 = pred2.detach().cpu()
		target2 = target2.detach().cpu()
		device = 'cpu'
	if torch.sum(target)==0:
		return float('nan')
	try:
		res =  torch.sum(pred2 + target)
		res = torch.tensor(zero_division,device=pred.device) if (res==0) else res
		res = 2*torch.sum(pred2 * target) / res
	except RuntimeError:
		if device != 'cpu':
			print('Error performing on GPU. retrying on CPU')
			res = dice(target=target, pred=pred2, threshold=None, compute_on_cpu=True, **args)
	return res

def torch_dice(target, pred, threshold=None, compute_on_cpu=False, **args):
	#print('Calculating dice')
	pred2 = pred.type(torch.float32) 
	target2 = target.type(torch.int32) 
	device = pred2.device
	if compute_on_cpu == True:
		pred2 = pred2.detach().cpu()
		target2 = target2.detach().cpu()
		device = 'cpu'
	pred2 = threshold_selector(pred2,threshold)
	pred2 = pred2.view(-1,1)
	target2 = target2.view(-1,1)
	#print(pred.device)
	dice = torchmetrics.Dice(**args).to(device=device)
	try:
		res = dice(preds=pred2, target=target2)
	except RuntimeError:
		if device != 'cpu':
			print('Error performing on GPU. retrying on CPU')
			res = torch_dice(target=target2, pred=pred2, threshold=threshold, compute_on_cpu=True, **args)
	return res

def torch_dice_loss(target, pred, **args):
	return {'loss': 1 - torch_dice(target, pred, **args)}

def dice_loss(target, pred, **args):
	return {'loss':1 -dice(target, pred, **args)}

def torch_tversky(target, pred, alpha=0.5, beta=0.5, zero_division=1e-4):
	true_pos  = torch.sum(target * pred)
	false_neg = torch.sum(target * (1 - pred))
	false_pos = torch.sum((1 - target) * pred)
	res = true_pos + alpha*false_neg + beta*false_pos
	res = torch.tensor(zero_division,device=pred.device) if (res==0) else res
	return true_pos / res

def torch_focal_tversky(target, pred, gamma=0.85, **tversky_param):
	res = torch_tversky(target, pred, **tversky_param)
	return torch.pow((1-res), gamma)

def torch_focal_tversky_loss(target, pred, **focal_tversky_param):
	res = torch_focal_tversky(target, pred, **focal_tversky_param)
	return {'loss': res}

def tversky(y_true, y_pred, alpha=0.5, beta=0.5, inverted=False):
	if not inverted:
		true_pos  = torch.sum(y_true * y_pred)
	else:
		true_pos  = torch.sum((1-y_true) * (1-y_pred))
	false_neg = torch.sum(y_true * (1 - y_pred))
	false_pos = torch.sum((1 - y_true) * y_pred)
	return true_pos / (true_pos + alpha*false_neg + beta*false_pos + 0.0000001)

def focal_tversky(y_true, y_pred, gamma=0.85, **tversky_param):
	res = tversky(y_true, y_pred, **tversky_param)
	return torch.pow((1-res), gamma)

def tversky_loss(y_true, y_pred, **tversky_param):
	return 1 - tversky(y_true,y_pred, **tversky_param)

def focal_tversky_loss(y_true, y_pred, **focal_tversky_param):
	res = focal_tversky(y_true,y_pred, **focal_tversky_param)
	return res

def kl_normal_loss(z_dist):
	# Calculate KL divergence
	loc = torch.tensor([0.0], dtype=z_dist.loc.dtype, device=z_dist.loc.device)
	scale =  torch.tensor([1.0], dtype=z_dist.scale.dtype, device=z_dist.scale.device)
	normal = dist.Normal(loc, scale)
	#assert not z_dist.loc.isnan().any()
	#assert not z_dist.scale.isnan().any()
	z_dist.scale = torch.clamp(z_dist.scale,None, 250)
	kl_values = dist.kl_divergence(z_dist, normal)
	#if kl_values.isnan().any():
	#    print(z_dist.loc[kl_values.isnan()])
	#    print(z_dist.scale[kl_values.isnan()])
	#    assert not kl_values.isnan().any()
		
	# Mean KL divergence across batch for each latent variable
	kl_means = torch.mean(kl_values, dim=0)
	#assert not kl_means.isnan().any()
		
	# KL loss is sum of mean KL of each latent variable
	kl_loss = torch.sum(kl_means)
	#assert not kl_loss.isnan().any()
	return kl_loss

def abs_diff(target, pred):
	return abs(target - pred)
	
def vqVAE_loss(target, pred, vq_loss, perplexity):
	rec_error = F.mse_loss(pred, target)
	return {'loss':rec_error + vq_loss, 'rec_error':rec_error, 'vq_loss':vq_loss, 'perplexity':perplexity}
   
   
def ceVAE_loss(target, pred, z_dist, y_ce_hat=None, ce_ratio=0.5, theta=1, beta=10000000 ):
	rec_loss_vae = F.l1_loss(pred, target)
	kl_loss = kl_normal_loss(z_dist)
	rec_loss_vae *= theta
	kl_loss      *= beta
	if y_ce_hat is not None:
		ce_loss = F.l1_loss(y_ce_hat, target)
		loss = (1 -ce_ratio)*(rec_loss_vae + kl_loss) + ce_ratio*ce_loss
		return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':kl_loss, 'ce_loss':ce_loss}
	else:
		loss = (1 -ce_ratio)*(rec_loss_vae + kl_loss)
		return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':kl_loss}
   
   
def disjoint_ceVAE_loss(target, pred, z_dist, mu, y_ce_hat=None, ce_ratio=0.5, theta=1, beta=10000000, cross_pow=1, gamma=1 ):
	rec_loss_vae = F.l1_loss(pred, target)
	kl_loss = kl_normal_loss(z_dist)
	#squeeze_dims = [2+x for x in range(len(mu.shape)-2)]
	#for dim in reversed(squeeze_dims):
	#    mu = torch.squeeze(mu,dim=dim)
	mu = mu.clone().detach().type(dtype=torch.float32)
	mu = mu.view(*mu.shape[0:2])
	#mu = torch.transpose(mu,0,1)
	pearson = torch.corrcoef(mu)
	pearson = torch.nan_to_num(pearson, nan=0.0)
	tri_pearson = torch.triu(pearson,diagonal=1).type(dtype=torch.float32)
	cross_samples = mu.shape[0] * (mu.shape[0]-1)//2
	cross_value = torch.sum(torch.abs(tri_pearson))/cross_samples if cross_samples!=0 else 0.
	mu_loss = torch.pow(cross_value,cross_pow) if cross_value!=0. else 0.
	rec_loss_vae *= theta
	kl_loss      *= beta
	mu_loss      *= gamma
	if y_ce_hat is not None:
		ce_loss = F.l1_loss(y_ce_hat, target)
		loss = (1 -ce_ratio)*(rec_loss_vae + kl_loss) + ce_ratio*ce_loss + mu_loss
		return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':kl_loss, 'ce_loss':ce_loss, 'mu_loss':mu_loss}
	else:
		loss = (1 -ce_ratio)*(rec_loss_vae + kl_loss) + mu_loss
		return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':kl_loss, 'mu_loss':mu_loss}
   
def ceVAE_with_gradient_loss(target, pred, y_hd_hat, z_dist, y_ce_hat=None, ce_ratio=0.5, theta=1, beta=10000000 ):
	
	with torch.set_grad_enabled(True):
		rec_loss_vae = F.l1_loss(y_hd_hat, target)
		kl_loss = kl_normal_loss(z_dist)
		comb_loss = rec_loss_vae + kl_loss
		comb_loss.backward()
		
		comb_loss = y_hd_hat.grad*F.l1_loss(pred, target)
		
	rec_loss_vae = F.l1_loss(pred, target)
	kl_loss = kl_normal_loss(z_dist)
	rec_loss_vae *= theta
	kl_loss      *= beta
	
	if y_ce_hat is not None:
		ce_loss = F.l1_loss(y_ce_hat, target)
		loss = (1 -ce_ratio)*(rec_loss_vae + kl_loss) + ce_ratio*ce_loss
		return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':kl_loss, 'ce_loss':ce_loss}
	else:
		loss = (1 -ce_ratio)*(rec_loss_vae + kl_loss)
		return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':kl_loss}
	
def fft_loss_fun(target, pred):
	assert target.shape[1] == 1, 'FFT-loss is not defined for more channels yet.'
	
	fft_target = torch.fft.fftn(target.type(torch.float32),norm="forward") 
	fft_target = fft_target.abs()
	fft_pred = torch.fft.fftn(pred.type(torch.float32),norm="forward") 
	fft_pred = fft_pred.abs()

	dims = len(fft_pred.shape[2:])
	cutoff = fft_pred.shape[2]//4
	assert fft_pred.shape[2]==fft_pred.shape[3], 'FFT-loss is not defined for tensors with different spatial dimentions'
	''''
	diff = abs(fft_pred-fft_target)

	loss=0
	if dims == 2:
		diff[:,:,1:-1,1:-1] = 0.
	else:
		diff[:,:,1:-1,1:-1,1:-1] = 0.
	
	loss = torch.sum(diff, dim=range(dims)+2)
	'''
	target_power = torch.sum(fft_target[:,:,cutoff:-cutoff,cutoff:-cutoff,cutoff:-cutoff], dim=list(range(2,2+dims)))
	pred_power   = torch.sum( fft_pred[:,:,cutoff:-cutoff,cutoff:-cutoff,cutoff:-cutoff],  dim=list(range(2,2+dims)))
	diff = torch.abs(pred_power - target_power)
	diff = torch.clamp(diff,torch.zeros_like(target_power), target_power)/target_power

	loss = torch.mean(diff)

	return loss
	
def fft_loss_fun2(target, pred):
	assert target.shape[1] == 1, 'FFT-loss is not defined for more channels yet.'
	
	fft_target = torch.fft.fftn(target.type(torch.float32),norm="forward") 
	fft_target = fft_target.abs()
	fft_pred = torch.fft.fftn(pred.type(torch.float32),norm="forward") 
	fft_pred = fft_pred.abs()

	dims = len(fft_pred.shape[2:])
	cutoff = fft_pred.shape[2]//4
	assert fft_pred.shape[2]==fft_pred.shape[3], 'FFT-loss is not defined for tensors with different spatial dimentions'
	''''
	diff = abs(fft_pred-fft_target)

	loss=0
	if dims == 2:
		diff[:,:,1:-1,1:-1] = 0.
	else:
		diff[:,:,1:-1,1:-1,1:-1] = 0.
	
	loss = torch.sum(diff, dim=range(dims)+2)
	'''
	target_power = torch.sum(fft_target[:,:,cutoff:-cutoff,cutoff:-cutoff,cutoff:-cutoff], dim=list(range(2,2+dims)))
	pred_power   = torch.sum( fft_pred[:,:,cutoff:-cutoff,cutoff:-cutoff,cutoff:-cutoff],  dim=list(range(2,2+dims)))
	diff = torch.abs(pred_power - target_power)
	diff = torch.clamp(diff,torch.zeros_like(target_power), target_power)/target_power

	loss = torch.mean(diff)

	return loss
 
def fft_loss_fun_zero_freq(target, pred):
	assert target.shape[1] == 1, 'FFT-loss is not defined for more channels yet.'
	
	fft_target = torch.fft.fftn(target.type(torch.float32),norm="forward") 
	fft_target = fft_target.abs()
	fft_pred = torch.fft.fftn(pred.type(torch.float32),norm="forward") 
	fft_pred = fft_pred.abs()

	dims = len(fft_pred.shape[2:])
	cutoff = fft_pred.shape[2]//4
	assert fft_pred.shape[2]==fft_pred.shape[3], 'FFT-loss is not defined for tensors with different spatial dimentions'
	diff = torch.abs(fft_pred-fft_target)
	
	target_power = fft_target[:,:,0,0,0]
	diff = torch.clamp(diff[:,:,0,0,0],torch.zeros_like(target_power), target_power)/target_power
	loss = torch.mean(diff)

	return loss

def loss_fun_zero_freq(target, pred):
	assert target.shape[1] == 1, 'loss_fun_zero_freq is not defined for more channels yet.'

	n_dims = len(target.shape[2:])
	diff = torch.abs(pred-target)
	
	mean_direction = [x for x in range(2,2+n_dims)]
	mean_direction = mean_direction[:]
	target_power = torch.mean(target,dim=mean_direction)
	pred_power   = torch.mean(pred,  dim=mean_direction)
	#print('target p: {}'.format(target_power[0,0]))
	#print('pred p: {}'.format(pred_power[0,0]))
	diff = torch.abs(target_power-pred_power)
	#diff = torch.clamp(diff,torch.zeros_like(target_power), target_power)/target_power
	loss = torch.mean(diff)

	return loss

def l1_loss(y_hat, y):
	return {'loss': F.l1_loss(y_hat, y)}
	
def hd_ceVAE_loss(target, pred, z_dist, y_hd_hat, y_ce_hat=None, ce_ratio=0.5, theta=1, beta=10000000, gamma=1 ):
	rec_loss_vae = F.l1_loss(pred, target)
	kl_loss = kl_normal_loss(z_dist)
	fft_loss = fft_loss_fun(target, y_hd_hat)
	fft_loss     *= gamma
	rec_loss_vae *= theta
	kl_loss      *= beta
	if y_ce_hat is not None:
		ce_loss = F.l1_loss(y_ce_hat, target)
		loss = (1 -ce_ratio)*(rec_loss_vae + kl_loss) + ce_ratio*ce_loss + fft_loss
		return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':kl_loss, 'fft_loss':fft_loss, 'ce_loss':ce_loss}
	else:
		loss = (1 -ce_ratio)*(rec_loss_vae + kl_loss) + fft_loss
		return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':kl_loss, 'fft_loss':fft_loss}
		
def VAE_loss(target, pred, z_dist, gamma=1e4):
	n_voxels = np.prod(target.shape[2:])
	rec_loss_vae = F.l1_loss(pred, target)
	#cross_entropy_loss = F.binary_cross_entropy(pred, target)
	#cross_entropy_loss = F.binary_cross_entropy_with_logits(pred,target)
	KL_loss = kl_normal_loss(z_dist)
	KL_loss *= gamma
	loss = rec_loss_vae + KL_loss
	return {'loss':loss, 'rec_loss_vae':rec_loss_vae, 'kl_loss':KL_loss}
	
def skipAE_loss(target,pred, alpha=1, beta=1e-7):
	p_loss = torch.nn.L1Loss()(target, pred)
	
	#gradient difference loss
	diff_kernel = torch.empty(2, device=target.device)
	diff_kernel[0] = 1
	diff_kernel[1] = -1
	gdl = torch.zeros(1, device=target.device)
	for i in range(3):
		view = [1,1,1]
		view[i]=-1
		df = diff_kernel.view([1,1]+view)
		if alpha != 1: #speedup trick
			gdl += torch.sum(torch.abs(F.conv3d(target, df))**alpha + torch.abs(F.conv3d(pred, df))**alpha)
		else:
			gdl += torch.sum(torch.abs(F.conv3d(target, df)) + torch.abs(F.conv3d(pred, df)))
	gdl *= beta
	loss = p_loss +gdl
	return {'loss':loss,'p_loss':p_loss,'gdl':gdl}

def gmVAE_loss(target, pred, w_mu, w_log_sigma, z_mu, z_log_sigma, z_wc_mus, z_wc_log_sigma_invs, pc, c_lambda=0.5, alpha=1, beta=1, gamma=1, **kwargs):
	dim_c = z_wc_mus.shape[1]
	z_mu_u_shape = [1 for x in z_wc_mus.shape]
	z_mu_u_shape[1] = dim_c
	
	# Build Losses
	# 1) Reconstruction Loss
	mean_p_loss = torch.nn.L1Loss()(target, pred)
	
	# 2. E_c_w[KL(q(z|x)|| p(z|w, c))]
	# calculate KL for each cluster
	# KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 ) here dim_c clusters, then we have batchsize * dim_z * dim_c
	# then [batchsize * dim_z* dim_c] * [batchsize * dim_c * 1]  = batchsize * dim_z * 1, squeeze it to batchsize * dim_z
	z_mu_u = z_mu.unsqueeze(dim = 1).repeat(z_mu_u_shape)
	z_logvar = z_log_sigma.unsqueeze(dim = 1).repeat(z_mu_u_shape)
	d_mu_2 = (z_mu_u - z_wc_mus) ** 2
	d_var = (torch.exp(z_logvar) + d_mu_2) * (torch.exp(z_wc_log_sigma_invs) + 1e-6)
	d_logvar = -1 * (z_wc_log_sigma_invs + z_logvar)
	kl = (d_var + d_logvar - 1) * 0.5
	con_prior_loss = torch.sum(torch.matmul(kl, pc.unsqueeze(dim = 2)).squeeze(dim = 2).view(target.shape[0], -1), dim = 1)
	mean_con_loss = torch.nanmean(con_prior_loss)

	# 3. KL(q(w|x)|| p(w) ~ N(0, I))
	# KL = 1/2 sum( mu^2 + var - logvar -1 )
	w_loss = 0.5 * torch.sum((torch.square(w_mu) + torch.exp(w_log_sigma) - w_log_sigma - 1).view(target.shape[0], -1), dim = 1)
	mean_w_loss = torch.nanmean(w_loss)

	# 4. KL(q(c|z)||p(c)) =  - sum_k q(k) log p(k)/q(k) , k = dim_c
	# let p(k) = 1/K#
	closs1 = torch.mul(pc, torch.log(pc * dim_c + 1e-8)).sum(dim = 1)
	#c_lambda = torch.full(closs1.shape, c_lambda, device=closs1.device)
	#c_loss = torch.maximum(closs1, c_lambda)
	c_loss = torch.clamp(closs1,min=c_lambda, max=None)
	dims = tuple( x for x in range(1,1+len(c_loss.shape[1:])))
	c_loss = c_loss.sum(dim = dims)
	mean_c_loss = torch.nanmean(c_loss)
	
	mean_con_loss *= alpha
	mean_w_loss   *= beta
	mean_c_loss   *= gamma

	loss = mean_p_loss + mean_con_loss + mean_w_loss + mean_c_loss
	return {'loss':loss, 'mean_p_loss':mean_p_loss, 'mean_con_loss':mean_con_loss, 'mean_w_loss':mean_w_loss, 'mean_c_loss':mean_c_loss}



def gmVAE3_loss(target, pred, w_mu, w_log_sigma, z_mu, z_log_sigma, z_wc_mus, z_wc_log_sigma_invs, pc, c_lambda=0.5, alpha=1, beta=1, gamma=1, theta=1, **kwargs):
	dim_c = z_wc_mus.shape[1]
	z_mu_u_shape = [1 for x in z_wc_mus.shape]
	z_mu_u_shape[1] = dim_c
	
	# Build Losses
	# 1) Reconstruction Loss
	mean_p_loss = torch.nn.L1Loss()(target, pred)
	
	# 2. E_c_w[KL(q(z|x)|| p(z|w, c))]
	# calculate KL for each cluster
	# KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 ) here dim_c clusters, then we have batchsize * dim_z * dim_c
	# then [batchsize * dim_z* dim_c] * [batchsize * dim_c * 1]  = batchsize * dim_z * 1, squeeze it to batchsize * dim_z
	z_mu_u = z_mu.unsqueeze(dim = 1).repeat(z_mu_u_shape)
	z_logvar = z_log_sigma.unsqueeze(dim = 1).repeat(z_mu_u_shape)
	d_mu_2 = (z_mu_u - z_wc_mus) ** 2
	d_var = (torch.exp(z_logvar) + d_mu_2) * (torch.exp(z_wc_log_sigma_invs) + 1e-6)
	d_logvar = -1 * (z_wc_log_sigma_invs + z_logvar)
	kl = (d_var + d_logvar - 1) * 0.5
	con_prior_loss = torch.sum(torch.matmul(kl, pc.unsqueeze(dim = 2)).squeeze(dim = 2).view(target.shape[0], -1), dim = 1)
	mean_con_loss = torch.mean(con_prior_loss)

	# 3. KL(q(w|x)|| p(w) ~ N(0, I))
	# KL = 1/2 sum( mu^2 + var - logvar -1 )
	w_loss = 0.5 * torch.sum((torch.square(w_mu) + torch.exp(w_log_sigma) - w_log_sigma - 1).view(target.shape[0], -1), dim = 1)
	mean_w_loss = torch.mean(w_loss)

	# 4. KL(q(c|z)||p(c)) =  - sum_k q(k) log p(k)/q(k) , k = dim_c
	# let p(k) = 1/K#
	closs1 = torch.mul(pc, torch.log(pc * dim_c + 1e-8)).sum(dim = 1)
	#c_lambda = torch.full(closs1.shape, c_lambda, device=closs1.device)
	#c_loss = torch.maximum(closs1, c_lambda)
	c_loss = torch.clamp(closs1,min=c_lambda, max=None)
	dims = tuple( x for x in range(1,1+len(c_loss.shape[1:])))
	c_loss = c_loss.sum(dim = dims)
	mean_c_loss = torch.mean(c_loss)

	
	fft_loss = loss_fun_zero_freq(target,pred)
	
	mean_con_loss *= alpha
	mean_w_loss   *= beta
	mean_c_loss   *= gamma
	fft_loss      *= theta

	loss = mean_p_loss + mean_con_loss + mean_w_loss + mean_c_loss + fft_loss
	return {'loss':loss, 'mean_p_loss':mean_p_loss, 'mean_con_loss':mean_con_loss, 'mean_w_loss':mean_w_loss, 'mean_c_loss':mean_c_loss, 'fft_loss':fft_loss}

def tpr_fpr(target,preds, threshold=None):
	#assert target.shape[1]==1, "tpr_fpr is not defined for multiclass inputs"
	pred = threshold_selector(preds, threshold)

	sum_directions = [x for x in range(0,len(target.shape))]
	sum_directions = sum_directions[:]

	tp = torch.sum(target*pred, dim=sum_directions)
	fn = torch.sum(target*(~pred), dim=sum_directions)
	fp = torch.sum((~target)*pred, dim=sum_directions)
	tn = torch.sum((~target)*(~pred), dim=sum_directions)

	tpr = torch.mean(tp/(tp+fn))
	fpr = torch.mean(fp/(fp+tn))

	return tpr, fpr

def calc_thresholds(pred, bins=10):
	#hist, ax = torch.histogram(pred[::1], bins=bins)
	pred += torch.randn_like(pred)*1e-6
	min_val = pred.min()
	max_val = pred.max()
	tmp_bins = 10000000
	hist = torch.histc(pred, bins=tmp_bins, min=min_val, max=max_val)
	step = (max_val-min_val)/tmp_bins
	#hist = hist.to(device=pred.device)
	ax   = torch.linspace(min_val-step/2, max_val+step/2, steps=tmp_bins+2, device=hist.device)
	hist = torch.cat([torch.tensor([0.], device=pred.device), hist, torch.tensor([0.], device=pred.device)])
	cumsum = torch.cumsum(hist, dim=0)
	levels = torch.linspace(cumsum.min(), cumsum.max(), bins)
	#ax_indices = []
	ax_list = []
	for l in levels:
		transf_l = torch.abs(cumsum-l)
		x = torch.argmin(transf_l).item()
		if cumsum[x]<l and x!=len(cumsum)-1 and cumsum[x+1]!=cumsum[x]:
			x_delta = (l-cumsum[x])/(cumsum[x+1]-cumsum[x])
			ax_list += [ax[x]+x_delta*(ax[x+1]-ax[x]),]
		elif cumsum[x]>l and x!=0 and cumsum[x]!=cumsum[x-1]:
			x_delta = (l-cumsum[x-1])/(cumsum[x]-cumsum[x-1])
			ax_list += [ax[x-1]+x_delta*(ax[x]-ax[x-1]),]
		else:
			ax_list += [ax[x],]
		#ax_indices += [torch.argmin(transf_l).item(),]
	#res = torch.cat([torch.tensor([0.], device=pred.device), ax[ax_indices], torch.tensor([1.], device=pred.device)])
	#res = torch.cat([torch.tensor([0.], device=pred.device), ax_list, torch.tensor([1.], device=pred.device)])
	return ax_list

def area_under_curve(x, y):
	'''
	x and y are python arrays
	'''
	area = 0.0
	for i in range(1, len(x)):
		area += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0
	return abs(area)
   
def recall_precision_fpr_tpr_f1(target, pred, threshold=None):
	preds = threshold_selector(pred, threshold)
	device = target.device
	zero = torch.zeros(size=(1,), device=device)

	sum_directions = [x for x in range(0,len(target.shape))]
	sum_directions = sum_directions[:]

	tp = torch.sum(target*preds, dim=sum_directions)
	fn = torch.sum(target*(~preds), dim=sum_directions)
	fp = torch.sum((~target)*preds, dim=sum_directions)
	tn = torch.sum((~target)*(~preds), dim=sum_directions)

	recall = tp / (tp + fn) if (tp + fn) > zero else zero
	precision = tp / (tp + fp) if (tp + fp) > zero else zero
	fpr = fp / (fp + tn) if (fp + tn) > zero else zero
	tpr = recall
	f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > zero else zero
	
	return recall.detach().cpu(), precision.detach().cpu(), fpr.detach().cpu(), tpr.detach().cpu(), f1.detach().cpu()

def pore_segmentation_curves__(orig_target, orig_pred, x, subsample=1, slices=None, bins=100, save_result_path=None, device=None):
	import imagetools.pore_segmentation as ps, imagetools.create_outer_mask as com
	device = orig_target.device if device is None else device
	
	target = orig_target[::subsample,::subsample,::subsample][::1].clone().detach()
	#target = target.detach().cpu()
	target = target.to(dtype=torch.bool)
	pred2  =   orig_pred[::subsample,::subsample,::subsample].clone().detach()
	thresholds = calc_thresholds(pred2, bins)
	#thresholds = np.linspace(-0.01,1+0.01,bins)
	m = dict()
	m['precision']  = []
	m['recall']     = []
	m['f1']         = []
	m['tpr']        = []
	m['fpr']        = []   
	m['threshold']  = []   
	if torch.is_tensor(x): #
		np_x = x[::subsample,::subsample,::subsample].detach().cpu().numpy() #
	else:
		np_x = x
	#if torch.is_tensor(thresholds): #
	#    thresholds = thresholds.detach().cpu().numpy() #
	obj_mask = com.create_outer_mask_core(np_x, threshold=0.3)
	obj_mask = torch.tensor(obj_mask,device=device, dtype=torch.bool)
	del np_x
	
	pred2 = torch.tensor(pred2,device=device)
	for i,thr in enumerate(thresholds):
		#if thr <= 0.:
		#    pred = torch.ones_like(pred2,device=device)
		#elif thr >= 1.:
		#    pred = torch.zeros_like(pred2,device=device)
		#else:
		#    pred = obj_mask*((pred2>thr))
		pred = obj_mask*((pred2>thr))
		pred = pred.to(dtype=torch.bool)
		recall, precision, fpr, tpr, f1 = recall_precision_fpr_tpr_f1(target, pred[::1], threshold=None)
		print('thr {} prec {} recall {}'.format(thr, precision,recall))

		m['precision'].append(precision.detach().cpu().item()) 
		m['recall'].append(recall.detach().cpu().item()) 
		m['f1'].append(f1) 
		m['tpr'].append(tpr.detach().cpu().item()) 
		m['fpr'].append(fpr.detach().cpu().item()) 
		m['threshold'].append(float(thr)) 
	m['auc']               = area_under_curve(m['fpr'],   m['tpr'])
	m['average_precision'] = area_under_curve(m['recall'],m['precision'])
	m['random_average_precision'] = torch.sum(target)/torch.numel(target)
	
	if save_result_path is not None:
		import pathlib, pickle, os
		print('Saving output in '+save_result_path)
		pathlib.Path(save_result_path).parent.absolute().mkdir(parents=True, exist_ok=True) 
		with open( save_result_path+'.pickle' , 'wb') as f:
			pickle.dump(m, f)
		m = m['average_precision']
					
	return m
				
def pore_segmentation_curves_(orig_target, orig_pred, x, subsample=1, slices=None, device=None, remove_small_objects=None, pred_mask=True, verbose=False, noise=False):
	import imagetools.pore_segmentation as ps, imagetools.create_outer_mask as com
	device = orig_target.device if device is None else device
	if slices is None or slices=='None':
		slices = [0,orig_target.shape[0],0,orig_target.shape[1],0,orig_target.shape[2]]
	else:
		tmp_slices = [0,orig_target.shape[0],0,orig_target.shape[1],0,orig_target.shape[2]]
		slices = [tmp_slices[i] if slices[i] is None else slices[i] for i in range(6)]
	target = orig_target[slices[0]:slices[1]:subsample,slices[2]:slices[3]:subsample,slices[4]:slices[5]:subsample].clone().detach()
	if target.device != device:
		target = target.detach().cpu().to(dtype=torch.bool, device=device)
	pred2  = orig_pred[slices[0]:slices[1]:subsample,slices[2]:slices[3]:subsample,slices[4]:slices[5]:subsample].clone().detach()
	if pred2.device != device:
		pred2  = pred2.detach().cpu().to(device=device)
	if noise:
		pred2 -= pred2.gt(0.9)*torch.abs(torch.randn_like(pred2))*1.e-2
		pred2 += pred2.lt(0.1)*torch.abs(torch.randn_like(pred2))*1.e-2
	pred2  = torch.clamp(pred2,0,1)
	
	m = dict()
	m['precision']  = []
	m['recall']     = []
	m['f1']         = []
	m['tpr']        = []
	m['fpr']        = []   
	m['threshold']  = []   
	
	#plt.imshow(pred2[:,:,200].detach().cpu())
	#plt.show()
 
	np_x = x[slices[0]:slices[1]:subsample,slices[2]:slices[3]:subsample,slices[4]:slices[5]:subsample].detach().cpu().numpy() if torch.is_tensor(x) else x
	obj_mask = com.create_outer_mask_core(np_x, threshold=0.4)
	obj_mask = torch.tensor(obj_mask, device=device, dtype=torch.bool)
	if pred_mask:
		np_x = pred2.detach().cpu().numpy() if torch.is_tensor(pred2) else pred2
		obj_mask *= torch.tensor(com.create_outer_mask_core(np_x, threshold=0.4, inverse=True), device=device, dtype=torch.bool) # intersection of the masks
	del np_x
	
	thr_step = 0.01
	current_threshold = -0.01
	current_recall = 1.
	current_precision = 0.
	current_fpr = 1.
	current_tpr = 1.
	min_delta_recall = 0.02
	min_delta_precision = 0.02
	min_delta_fpr = 0.02
	min_delta_tpr = 0.02
	recall=1.
	step_cut=1
	
	if remove_small_objects is not None:
		import imagetools.extract_connected_cells as ecc
		number_of_positives=target.sum().detach().cpu()
	
	
	while recall > 0:
		thr = float(current_threshold + thr_step/(2**(step_cut-1)))
		if thr < 0.:
			thr=0.
		while thr>1.0:
			step_cut+=1
			thr = float(current_threshold + thr_step/(2**(step_cut-1)))
		
		if thr <= 0.:
			#pred = torch.ones_like(pred2,device=device)
			pred = obj_mask
		elif thr >= 1.:
			pred = torch.zeros_like(pred2,device=device)
		else:
			pred = obj_mask*(pred2>thr)
		#pred = obj_mask*(pred2>thr)
		pred = pred.to(dtype=torch.bool)
		
		if remove_small_objects is not None and pred.sum().detach().cpu() < 4*number_of_positives:
			torch.cuda.empty_cache()
			print('removing small objects') if verbose else None
			pred = ecc.extract_connected_cells_and_remove_small_clusters(min_dimension=remove_small_objects, input=pred, device=device)
			#pass
		recall, precision, fpr, tpr, f1 = recall_precision_fpr_tpr_f1(target, pred[::1], threshold=None)
		print('thr {} prec {} recall {}'.format(thr, precision,recall)) if verbose else None

		if (abs(recall - current_recall) < min_delta_recall and abs(precision - current_precision) < min_delta_precision) or step_cut>40:
			m['precision'].append(float(precision)) 
			m['recall'].append(float(recall)) 
			m['f1'].append(float(f1)) 
			m['tpr'].append(float(tpr)) 
			m['fpr'].append(float(fpr)) 
			m['threshold'].append(float(thr)) 
			print(len(m['precision'])) if verbose else None
			current_recall = recall
			current_precision = precision
			current_threshold = thr
			step_cut=1
		else:
			step_cut+=1
			recall = current_recall
	m['threshold'].append(1.0)
	m['f1'].append(0.0)
	m['recall'].append(0.0)
	m['precision'].append(0.0)
	m['tpr'].append(0.0)
	m['fpr'].append(0.0)
				
	m['auc']               = area_under_curve(m['fpr'],   m['tpr'])
	m['average_precision'] = area_under_curve(m['recall'],m['precision'])
	m['random_average_precision'] = (torch.sum(target)/torch.numel(target)).detach().cpu().numpy()
	
	return m

def find_boundary(vol):
	dir_x = vol.sum(dim=[1,2])
	dir_y = vol.sum(dim=[0,2])
	dir_z = vol.sum(dim=[0,1])
	# find the index of the first and last non-zero element in dir_x
	x_first, x_last = torch.nonzero(dir_x, as_tuple=True)[0][0], torch.nonzero(dir_x, as_tuple=True)[0][-1]
	y_first, y_last = torch.nonzero(dir_y, as_tuple=True)[0][0], torch.nonzero(dir_y, as_tuple=True)[0][-1]
	z_first, z_last = torch.nonzero(dir_z, as_tuple=True)[0][0], torch.nonzero(dir_z, as_tuple=True)[0][-1]
	return x_first, x_last, y_first, y_last, z_first, z_last

def pore_segmentation_curves(orig_target, orig_pred, x, subsample=1, slices=None, device=None, remove_small_objects=None, pred_mask=True, verbose=False, noise=False):
	import imagetools.pore_segmentation as ps, imagetools.create_outer_mask as com
	device = orig_target.device if device is None else device
	if slices is None or slices=='None':
		slices = [0,orig_target.shape[0],0,orig_target.shape[1],0,orig_target.shape[2]]
	else:
		tmp_slices = [0,orig_target.shape[0],0,orig_target.shape[1],0,orig_target.shape[2]]
		slices = [tmp_slices[i] if slices[i] is None else slices[i] for i in range(6)]
	target = orig_target[slices[0]:slices[1]:subsample,slices[2]:slices[3]:subsample,slices[4]:slices[5]:subsample].clone().detach()
	if target.device != device:
		target = target.detach().cpu().to(dtype=torch.bool, device=device)
	pred2  = orig_pred[slices[0]:slices[1]:subsample,slices[2]:slices[3]:subsample,slices[4]:slices[5]:subsample].clone().detach()
	if pred2.device != device:
		pred2  = pred2.detach().cpu().to(device=device)
	if noise:
		pred2 -= pred2.gt(0.9)*torch.abs(torch.randn_like(pred2))*1.e-2
		pred2 += pred2.lt(0.2)*torch.abs(torch.randn_like(pred2))*1.e-2
	pred2  = torch.clamp(pred2,0,1)
	
	m = dict()
	m['precision']  = []
	m['recall']     = []
	m['f1']         = []
	m['tpr']        = []
	m['fpr']        = []   
	m['threshold']  = []   
	
	#plt.imshow(pred2[:,:,200].detach().cpu())
	#plt.show()
 
	np_x = x[slices[0]:slices[1]:subsample,slices[2]:slices[3]:subsample,slices[4]:slices[5]:subsample].detach().cpu().numpy() if torch.is_tensor(x) else x[slices[0]:slices[1]:subsample,slices[2]:slices[3]:subsample,slices[4]:slices[5]:subsample]
	np_x = np.copy(np_x)
	obj_mask = com.create_outer_mask_core(np_x, threshold=0.4)
	obj_mask = torch.tensor(obj_mask, device=device, dtype=torch.bool)
	#x_min, x_max, y_min, y_max, z_min, z_max = find_boundary(obj_mask)
	#target = target[x_min:x_max,y_min:y_max,z_min:z_max]
	#pred2  = pred2[x_min:x_max,y_min:y_max,z_min:z_max]
	#obj_mask = obj_mask[x_min:x_max,y_min:y_max,z_min:z_max]
	
	if pred_mask:
		np_x = pred2.detach().cpu().numpy() if torch.is_tensor(pred2) else pred2
		np_x = np.copy(np_x)
		obj_mask *= torch.tensor(com.create_outer_mask_core(np_x, threshold=0.4, inverse=True), device=device, dtype=torch.bool) # intersection of the masks
	del np_x
	
	thr_step = 0.01
	current_threshold = -0.01
	current_recall = 1.
	current_precision = 0.
	current_fpr = 1.
	current_tpr = 1.
	min_delta_recall = 0.04
	min_delta_precision = 0.04
	min_delta_fpr = 0.04
	min_delta_tpr = 0.04
	recall=1.
	step_cut=1
	is_first_step = True
	
	while recall > 0 and current_threshold<1-1.e-6:
		thr = float(current_threshold + thr_step/(2**(step_cut-1)))
		if thr < 0.:
			thr=0.
		while thr>1.0:
			step_cut+=1
			thr = float(current_threshold + thr_step/(2**(step_cut-1)))
		
		pred = obj_mask*(pred2>=thr)
		pred = pred.to(dtype=torch.bool)
		
		recall, precision, fpr, tpr, f1 = recall_precision_fpr_tpr_f1(target, pred, threshold=None)
		print('thr {} prec {} recall {}'.format(thr, precision,recall)) if verbose else None
  
		if is_first_step:
			current_fpr = fpr
			current_tpr = tpr
			m['threshold'].append(0.0)
			m['f1'].append(0.0)
			m['recall'].append(1.0)
			m['precision'].append(0.0)
			m['tpr'].append(1.0)
			m['fpr'].append(1.0)
			is_first_step = False

		if (abs(recall-current_recall)<min_delta_recall and abs(precision-current_precision)<min_delta_precision and abs(fpr-current_fpr)<min_delta_fpr and abs(tpr-current_tpr)<min_delta_tpr) or step_cut>40:
			m['precision'].append(float(precision)) 
			m['recall'].append(float(recall)) 
			m['f1'].append(float(f1)) 
			m['tpr'].append(float(tpr)) 
			m['fpr'].append(float(fpr)) 
			m['threshold'].append(float(thr)) 
			print(len(m['precision'])) if verbose else None
			current_recall = recall
			current_precision = precision
			current_threshold = thr
			current_fpr = fpr
			current_tpr = tpr
			step_cut=1
		else:
			step_cut+=1
			recall = current_recall
   
	# calculate the nearest point of the fpr-tpr curve to the point (1,0) 
	min_dist_pr  = 2.0
	min_dist_roc = 2.0
	for thr, curr_fpr, curr_tpr, curr_prec, curr_recall in zip(m['threshold'], m['fpr'], m['tpr'],  m['precision'], m['recall']):
		dist = np.sqrt(curr_fpr**2 + (1-curr_tpr)**2)
		if dist < min_dist_roc:
			min_dist_roc = dist
			nearest_point_roc = (curr_fpr, curr_tpr, thr)
		dist = np.sqrt((1-curr_prec)**2 + (1-curr_recall)**2)
		if dist < min_dist_pr:
			min_dist_pr = dist
			nearest_point_pr = (curr_prec, curr_recall, thr)
 
	m['nearest_point_pr']  = nearest_point_pr
	m['nearest_point_roc'] = nearest_point_roc
	print('Nearest point (pr) is {} at threshold {}'.format(nearest_point_pr[0:2], nearest_point_pr[2]))
	print('Nearest point (ROC) is {} at threshold {}'.format(nearest_point_roc[0:2], nearest_point_roc[2]))
 
	m['threshold'].append(1.0)
	m['f1'].append(0.0)
	m['recall'].append(0.0)
	m['precision'].append(0.0)
	m['tpr'].append(0.0)
	m['fpr'].append(0.0)
				
	m['auc']               = area_under_curve(m['fpr'],   m['tpr'])
	m['average_precision'] = area_under_curve(m['recall'],m['precision'])
	m['random_average_precision'] = (torch.sum(target)/torch.numel(target)).detach().cpu().numpy()
	
	return m
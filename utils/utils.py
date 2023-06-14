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


from skimage import filters
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch, torchvision, gc, pynvml

class allocateMemory:
    dummy_memory = None
    gpu_index = None
    
    def __init__(self, gpu_index):
        pynvml.nvmlInit()
        self.allocate(gpu_index)
    
    def allocate(self, gpu_index):
        if self.gpu_index is None:
            self.gpu_index = gpu_index
        else:
            raise Exception('Already used for another gpu.. please do not mix things up')
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        if self.dummy_memory is None:
            self.dummy_memory = torch.cuda.ByteTensor(mem_info.free)
        
    def deallocate(self):
        if self.dummy_memory is None:
            self.dummy_memory = None
            

def apply_input_mask_gpu(pars, y_hat, x, compute_on_gpu=True, **kwargs):
    low, up = pars
    
    tensor_mask1 = tensor_mask2 = torch.ones_like(x)
    if low is not None:
        tensor_mask1 = 1.0*(x>low)
    if up is not None:
        tensor_mask2 = 1.0*(x<up)
    mask = torch.logical_and(tensor_mask1, tensor_mask2).type(torch.float32)
    
    return y_hat*mask

def apply_input_mask_hook(pars, gpu_list, y_hat, x, compute_on_gpu=True):
    low, up = pars
    
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y_hat):
        y_hat     = torch.tensor(y_hat)

    if compute_on_gpu:
        x2     = x.to(device=gpu_list['x'])
        y_hat2 = y_hat.to(device=gpu_list['y_hat'])
    else:
        x2     = x.to(device='cpu')
        y_hat2 = y_hat.to(device='cpu')
    
    try:
        res = apply_input_mask_gpu(pars, y_hat2, x2)
        output={}
        output['y_hat'] = res
    except RuntimeError:
        print('apply_input_mask_hook: Too large for GPU, calculating on CPU')
        res = apply_input_mask_gpu(pars, y_hat, x, compute_on_gpu=False)
    return output

from skimage import filters

def apply_input_mask_and_otsu_threshold(x, y_hat, pars, gpu):
    low, up = pars
    
    tensor_mask1 = tensor_mask2 = np.ones_like(x)
    if low is not None:
        tensor_mask1 = 1.0*(x>low)
    if up is not None:
        tensor_mask2 = 1.0*(x<up)
    mask = np.logical_and(tensor_mask1, tensor_mask2)
    
    res = filters.threshold_otsu(y_hat*mask)
    
    return torch.tensor((y_hat>res), device=gpu)
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


import numpy as np
import random 

def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value
    
    
    
def delete_parallelepipedic_regions( x, rnd_size=None, rnd_multiplicity=[1, 3], rnd_value=0, channelwise_deletion=False ):
    
    batch_size = x.shape[0]
    dims = x.shape[2:]
    dim_list = range(len(dims))
    
    for sample_idx in range(batch_size):
        if isinstance(rnd_size, (list, tuple, np.ndarray)):
            if len(rnd_size)==2*len(dims):
                size = [get_range_val(rnd_size[i]) for i in dim_list]
            elif len(rnd_size)==2:
                size = [get_range_val(rnd_size) for i in dim_list]
            else:
                raise Exception ('check the rnd_size shape. Should be a list of 2 elements or 2 times the number of dimensions')
        else:
            raise Exception ('rnd_size argument should be a list, tuple or np.array')
        if isinstance(rnd_multiplicity, (list, tuple, np.ndarray)):
            multiplicity = get_range_val(rnd_multiplicity)
        else:
            multiplicity = rnd_multiplicity
            

        noisy_x = mask_random_parallelepipeds(
            x,
            sample_idx=sample_idx,
            size=size,
            multiplicity=multiplicity,
            rnd_value=rnd_value,
            channelwise_deletion=channelwise_deletion
        )
    
    return noisy_x

def mask_random_parallelepipeds(x, sample_idx, size, multiplicity, rnd_value, channelwise_deletion=False):

    dims = list(x.shape[2:])
    n_channels = x.shape[1]
    #size = torch.Size(size)

    for current_box in range(multiplicity):
        start = []
        for i in range(len(dims)):
            start.append(random.randint(0, dims[i]-size[i]))
        
        if not channelwise_deletion:
            if isinstance(rnd_value, (list, tuple, np.ndarray)):
                value = get_range_val(rnd_value)
            else:
                value= rnd_value
            if len(dims)==2:
                x[sample_idx, :, start[0]:(start[0]+size[0]), start[1]:(start[1]+size[1]) ] = value
            else:
                x[sample_idx, :, start[0]:(start[0]+size[0]), start[1]:(start[1]+size[1]), start[2]:(start[2]+size[2]) ] = value
                #print('{} {} {} {} {} {}'.format(start[1],(start[0]+size[0]), start[1],(start[1]+size[1]), start[2],(start[2]+size[2])))
                #print(value)
                #print(x[0, 0, start[0]+size[0]-1,start[1]+ size[1]-1, start[2]+size[2]-1 ])
                #print(x[0,0,0,0,0])
        else:
            for channel in range(n_channels):
                if isinstance(rnd_value, (list, tuple, np.ndarray)):
                    current_value = get_range_val(rnd_value)
                else:
                    current_value = rnd_value
                if len(dims)==2:
                    x[sample_idx, channel, start[0]:(start[0]+size[0]), start[1]:(start[1]+size[1]) ] = current_value
                else:
                    x[sample_idx, channel, start[0]:(start[0]+size[0]), start[1]:(start[1]+size[1]), start[2]:(start[2]+size[2]) ] = current_value

    return x

import torch, time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    x = torch.ones((64,64,64)).unsqueeze(0).unsqueeze(0)
    noisy_x = delete_parallelepipedic_regions( x, rnd_size=(0, np.amax(x.shape)// 2), rnd_multiplicity=(1, 3), rnd_value=0 )
    
    for idx in range(noisy_x.shape[4]):
        #plt.figure()
        #plt.imshow(noisy_x[0,0,:,:,idx*16])
        #plt.show(block=False)
        #time.sleep(5)
        from PIL import Image
        data = noisy_x[0,0,:,:,idx].cpu().detach().numpy()
        im = Image.fromarray(data)
        im = im.convert('RGB')
        im.save('/home/diuso/tmp/file_'+str(idx)+'.jpeg')
        
        
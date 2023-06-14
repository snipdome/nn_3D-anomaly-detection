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


import pynvml, gc, os, glob, json

def list_str_to_int(values):
    pieces = values.split(',')
    int_pieces = [int(piece) for piece in pieces]
    return int_pieces

def find_free_gpu(sorted_preference=None):
    
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    if sorted_preference is None:
        sorted_preference = range(deviceCount)
    
    for gpu_index in sorted_preference:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #print('videocard {} has {} free memory and {} total memory (free {}%)'.format(gpu_index,mem_info.free, mem_info.total,mem_info.free/mem_info.total))
        if (mem_info.free / mem_info.total) > 0.95:
            print('Selected video-card {} in auto-mode'.format(gpu_index))
            return [gpu_index]
    raise Exception('Unable to find a free gpu in automatic mode')

def find_free_gpus(tolerance=0.05):
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    cuda_gpu_list = range(deviceCount)
    free_gpus = []
    gpu_string = ''
    
    for gpu_index in cuda_gpu_list:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #print('videocard {} has {} free memory and {} total memory (free {}%)'.format(gpu_index,mem_info.free, mem_info.total,mem_info.free/mem_info.total))
        if (mem_info.free / mem_info.total) > 1-tolerance:
            if len(free_gpus)==0:
                gpu_string = str(gpu_index)
            else:
                gpu_string += ','+str(gpu_index)
            free_gpus.append(gpu_index)
    
    CUDA_VISIBLE_DEVICES = gpu_string
    #print(CUDA_VISIBLE_DEVICES)
    if len(free_gpus) == 0:
        raise Exception('Unable to find a free gpu in automatic mode')
    return CUDA_VISIBLE_DEVICES

            
def set_gpu_flags(skip_gpus=None, use_gpus=None):
    if skip_gpus is not None:
        for gpu_to_avoid in skip_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].replace(str(gpu_to_avoid)+',', '')
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].replace(str(gpu_to_avoid), '') # if was last element
        if os.environ['CUDA_VISIBLE_DEVICES'].endswith(','):
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'][:-1]
        assert os.environ['CUDA_VISIBLE_DEVICES'] != 0, 'No available CUDA gpu in the machine.'
        
    if use_gpus is not None:
        gpu_str = ''
        for i, gpu in enumerate(use_gpus):
            if i!=0:
                gpu_str = gpu_str+','
            gpu_str = gpu_str + str(gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

    print('The CUDA visible devices have index '+os.environ['CUDA_VISIBLE_DEVICES']) 
    

def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.

    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
        
    try: # check if could be interpreted as a number
        value2 = float(value)
        
        if value2 == value.strip('.'):
            value = int(value2)
        else:
            value = value2
    except ValueError:
        pass
    
    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    #print(items)
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

def add_branch(tree, vector, value):
    """
    Given a dict, a vector, and a value, insert the value into the dict
    at the tree leaf specified by the vector.  Recursive!

    Params:
        data (dict): The data structure to insert the vector into.
        vector (list): A list of values representing the path to the leaf node.
        value (object): The object to be inserted at the leaf

    Example 1:
    tree = {'a': 'apple'}
    vector = ['b', 'c', 'd']
    value = 'dog'

    tree = add_branch(tree, vector, value)

    Returns:
        tree = { 'a': 'apple', 'b': { 'c': {'d': 'dog'}}}

    Example 2:
    vector2 = ['b', 'c', 'e']
    value2 = 'egg'

    tree = add_branch(tree, vector2, value2)    

    Returns:
        tree = { 'a': 'apple', 'b': { 'c': {'d': 'dog', 'e': 'egg'}}}

    Returns:
        dict: The dict with the value placed at the path specified.

    Algorithm:
        If we're at the leaf, add it as key/value to the tree
        Else: If the subtree doesn't exist, create it.
              Recurse with the subtree and the left shifted vector.
        Return the tree.

    """
    key = vector[0]
    tree[key] = value \
        if len(vector) == 1 \
        else add_branch(tree[key] if key in tree else {},
                        vector[1:],
                        value)
    return tree

def parse_string_dict_to_structured_dict(string_dict):
    structured_dict = {}
    for key,value in string_dict.items():
        split_key = key.split('.')
        add_branch(structured_dict, split_key, value)
    #print(structured_dict)
    return structured_dict

import collections.abc

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        elif k.endswith(']'):
            index = int(k[k.index('[')+1:-1])
            key = k[:k.index('[')]
            if not isinstance(d[key], list):
                d[key] = []
            if len(d[key]) <= index:
                d[key].extend([None]*(index-len(d[key])+1))
                d[key][index] = v
        else:
            if v=="null":
                d[k] = None
            elif isinstance(v,str) and v.startswith('[') and v.endswith(']'):
                tmp = v[1:-1].replace(' ','').split(',')
                for i in range(len(tmp)):
                    tmp[i] = int(tmp[i]) if tmp[i] != 'None' else None
                d[k] = tmp
            else:
                d[k] = v
    return d
            
def find_last_checkpoint(checkpoint_general_path, model_name, version):
    
    template_name = model_name +"-*-last-" +version +'.ckpt'
    files = glob.glob( os.path.join(checkpoint_general_path, template_name) )
    if len(files) == 0:
        raise Exception('No suitable checkpoint has been found. Please provide a path to the checkpoint manually.')
    elif len(files) > 1:
        print('These checkpoints have been found:')
        for ckpt_file in files:
            print(ckpt_file)
        raise Exception('Too many checkpoints has been found. Please provide a path one of them manually.')
    else:
        print('The following checkpoint has been found: '+files[0])
        return files[0]
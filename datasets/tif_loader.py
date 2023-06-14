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
import torchio as tio
import torch
from torch.utils.data import Dataset
from torchio.data.subject import Subject
import glob, os
from .utils import extractSliceNumber


def getInverseProbabilityMap(data):
    n_voxels = torch.numel(data)
    positives = torch.count_nonzero(data)
    negatives = n_voxels-positives
    likelihood = (data!=0.)*negatives/n_voxels + (data==0.)*positives/n_voxels
    return likelihood

def getListofFilesPerSubject(global_path):
    subj_list={}
    for path in glob.glob(global_path): # For all the file/folders that are given by the regex
        if os.path.isdir(path):       # If it is indeed a folder
            #print('path is {}'.format(path))
            for subfolders in [x[1] for x in sorted(os.walk(path,followlinks=True))]+['.']: # Get the subfolders of each folder
                if subfolders != []:
                    for subfolder in subfolders:
                        #print('subfolder is {}'.format(os.path.join(path,subfolder)))
                        file_list = sorted(glob.glob(os.path.join(path,subfolder, '*.tif')),  key=extractSliceNumber)
                        if file_list == []:
                            file_list = sorted(glob.glob(os.path.join(path,subfolder, '*.tiff')),  key=extractSliceNumber)
                        if file_list != []:
                            #print('Found subject {} with {} elements'.format(subfolder, len(file_list)))
                            subj_list[subfolder] = file_list
    return subj_list

def checkConsistencyDatasets(data_subj, label_subj ):
    for key in data_subj.keys():
        if key not in label_subj.keys():
            print(data_subj.keys())
            print(label_subj.keys())
            raise Exception('The data set and the label/sampling set have different subjects names (unable to find subject {}).'.format(key))
        elif len(data_subj[key]) != len(label_subj[key]):
            raise Exception('The subject {} has different amount of slices in the data set ({}) and label/sampling set ({}).'.format(key,len(data_subj[key]),len(label_subj[key])))
        elif data_subj[key] == 0:
            raise Exception('The subject {} contains no slices.'.format(key))
    return
            
class tifLoadDataset(Dataset):
    def __init__(self,  path, label=None, sampling_likelihood=None, sampling=None, lazy=False, **kwargs):
        ''' 
        
        :param sampling: It is used only to tell the loader if it should add an additional tag to the subject, in which it is stored
            the information about the sampling likelihood. Values: uniform, grid, equal_chance
        :type sampling: str 
        '''
        self.data_subj  = getListofFilesPerSubject(path)
        if label is not None:
            self.label_subj = getListofFilesPerSubject(label)
            checkConsistencyDatasets( self.data_subj, self.label_subj )
        if sampling_likelihood is not None:
            self.sampl_subj = getListofFilesPerSubject(sampling_likelihood)
            checkConsistencyDatasets( self.data_subj, self.sampl_subj )
        
        print('Loaded {} subjects'.format(len(self.data_subj)))

        self.datasamples = []
        for subj in self.data_subj.keys():
            images = {}
            images['img']     = tio.ScalarImage(self.data_subj[subj])
            if not lazy:
                img           = torch.transpose(images['img'].tensor,3,0)
                images['img'] = tio.ScalarImage(tensor=img,check_nans=False)
            if label is not None:
                images['label']     = tio.LabelMap(self.label_subj[subj])
                if not lazy:
                    lbl           = torch.transpose(images['label'].tensor,3,0)
                    images['label'] = tio.LabelMap(tensor=lbl,check_nans=False)
            if sampling_likelihood is not None:
                images['sampling_likelihood']     = tio.LabelMap(self.sampl_subj[subj])
                if not lazy:
                    sampling_l = torch.transpose(images['sampling_likelihood'].tensor,3,0)
                    images['sampling_likelihood'] = tio.LabelMap(tensor=sampling_l,check_nans=False)
            elif sampling == 'equal_chance':
                sampling_l = getInverseProbabilityMap(lbl)
                images['sampling_likelihood']     = tio.LabelMap(tensor=sampling_l,check_nans=False)
            
            subject = Subject( **images )
            self.datasamples.append(subject)
                

    def __len__(self):
        return len(self.datasamples)

    def __getitem__(self, item):

        if self.lazypatch:
            data   = []
            labels = []
            for slice_idx in range(len(self.data_files[item])):
                y = np.expand_dims(tio.ScalarImage(self.data_files[item][slice_idx]), axis=2).astype(dtype=np.float32)
                data  = (y if slice_idx == 0 else np.concatenate((data,y),axis=2))
                z = np.expand_dims(tio.LabelMap(self.label_files[item][slice_idx] ), axis=2).astype(dtype=np.float32)
                labels = (z if slice_idx == 0 else np.concatenate((labels,z),axis=2))
            if self.torchiosub:
                subject = Subject({'img': y, 'label':z } )
                if self.transform is not None:
                    subject = self.transform(subject)

                return subject
            else:
                # how to apply the transform in this case?
                return {'img': data, 'label':labels }
        else:
            if self.torchiosub:
                subject = self.datasamples[item]  
                if self.transform is not None:
                    subject = self.transform(subject)
                #if item == 0:
                #    print("rank "+str(torch.distributed.get_rank())+" value of first pixel of first sample of dataloader: "+str(subject['img']['data'][0,0,0,0]))
                return subject
            else:
                # how to apply the transform in this case?
                return {'img': data, 'label':labels }

    
    def set_transform(self, transform) -> None:
        """Set the :attr:`transform` attribute.

        Args:
            transform: Callable object, typically an subclass of
                :class:`torchio.transforms.Transform`.
        """
        if transform is not None and not callable(transform):
            message = (
                'The transform must be a callable object,'
                f' but it has type {type(transform)}'
            )
            raise ValueError(message)
        self._transform = transform
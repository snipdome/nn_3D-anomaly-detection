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


import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from datasets import tifLoadDataset
import torchio as tio
import torch, os
from .data_augmentation import RandomRescale, RandomWave

class VolumetricDataLoader(pl.LightningDataModule):
    def __init__(self, training, validation, test_and_predict=None, num_workers=0, patch_size=None, is_distributed=False, file_format='tif', data_augm_v=None, aggregate_patches=False, **kwarg):
        super().__init__()
        self.training               = training
        self.validation             = validation
        self.test_and_predict       = test_and_predict
        self.test                   = kwarg.get('test')    if kwarg.get('test')    is not None else {}
        self.predict                = kwarg.get('predict') if kwarg.get('predict') is not None else {}
        self.num_workers            = num_workers
        self.is_distributed         = is_distributed
        self.patch_size             = patch_size
        self.file_format            = file_format
        self.data_augm_v            = data_augm_v
        self.aggregate_patches      = aggregate_patches

    def setup(self, stage: str = None):
        transform = self.get_transform(self.data_augm_v)
        if stage == "fit" or stage is None:    
            trainset = tifLoadDataset(
                **self.training['dataset'],
                sampling=self.training['sampling']
                ) 
            self.trainset = tio.data.SubjectsDataset(trainset.datasamples, transform=transform)

            if self.training['sampling'] == 'equal_chance': # equal likelihood of sampling on black labels and white labels
                sampler = tio.data.WeightedSampler(patch_size=self.patch_size, probability_map='sampling_likelihood')
            elif self.training['sampling'] in ('label','sampling_likelihood'):
                sampler = tio.data.WeightedSampler(patch_size=self.patch_size, probability_map=self.training['sampling'] )
            else:
                sampler = tio.data.UniformSampler(patch_size=self.patch_size)

            if self.patch_size:
                self.trainset = tio.data.Queue(
                    subjects_dataset = self.trainset,
                    max_length = self.training['queue_length'],
                    shuffle_subjects = True,
                    shuffle_patches = True,
                    samples_per_volume = self.training['patches_per_volume'],
                    sampler = sampler,
                    num_workers = self.num_workers)

            if self.validation['sampling'] == 'equal_chance': # equal likelihood of sampling on black labels and white labels
                sampler = tio.data.WeightedSampler(patch_size=self.patch_size, probability_map='sampling_likelihood')
            elif self.validation['sampling'] in ('label','sampling_likelihood'):
                sampler = tio.data.WeightedSampler(patch_size=self.patch_size, probability_map=self.validation['sampling'])
            elif self.validation['sampling'] =='uniform':
                sampler = tio.data.UniformSampler(patch_size=self.patch_size)
            else: 
                self.grid_sampler = tio.inference.GridSampler(
                    patch_size    = self.patch_size,
                    patch_overlap = self.validation['patch_overlap'])#,
                    #padding_mode = 'edge')
                self.grid_sampler.padding_mode = 'edge'
            valset   = tifLoadDataset(
                **self.validation['dataset'],  
                sampling=self.validation['sampling']) 
            
            val_transform = self.get_transform(self.validation.get('data_augm_v',None))
            self.valset = tio.data.SubjectsDataset(valset.datasamples, transform=val_transform)
            if self.validation['sampling'] == 'grid':
                self.valset = tio.data.Queue(
                    subjects_dataset = self.valset,
                    max_length = self.validation['queue_length'],
                    shuffle_subjects = False,
                    shuffle_patches = False,
                    samples_per_volume = self.validation['patches_per_volume'],
                    sampler = self.grid_sampler,
                    num_workers = self.num_workers
                )
            else:
                self.valset = tio.data.Queue(
                    subjects_dataset = self.valset,
                    max_length = self.validation['queue_length'],
                    shuffle_subjects = True,
                    shuffle_patches = True,
                    samples_per_volume = self.validation['patches_per_volume'],
                    sampler = sampler,
                    num_workers = self.num_workers
                )

        elif stage == 'test':
            testset  = tifLoadDataset(self.test_and_predict['dataset']['path'], self.test_and_predict['dataset']['label']) 
            self.testset = tio.data.SubjectsDataset(testset.datasamples)
            if self.patch_size:
                self.grid_sampler = tio.inference.GridSampler(
                    subject = self.testset[0],
                    patch_size    = self.patch_size,
                    patch_overlap = self.test_and_predict['patch_overlap'],
                    padding_mode = 'edge'
                )
                
                self.testset = tio.data.Queue(
                    subjects_dataset = self.testset,
                    max_length = len(self.grid_sampler),
                    shuffle_subjects = False,
                    shuffle_patches = False,
                    samples_per_volume = len(self.grid_sampler),
                    sampler = self.grid_sampler,
                    num_workers = self.num_workers
                )   

        elif stage == 'predict':
            predset  = tifLoadDataset(self.test_and_predict['dataset']['path']) 
            self.predset = tio.data.SubjectsDataset(predset.datasamples)
            if self.patch_size:
                self.grid_sampler = tio.inference.GridSampler(
                    subject = self.predset[0],
                    patch_size    = self.patch_size,
                    patch_overlap = self.test_and_predict['patch_overlap'],
                    padding_mode = 'edge'
                )
                self.predset = tio.data.Queue(
                    subjects_dataset = self.predset,
                    max_length = len(self.grid_sampler),
                    shuffle_subjects = False,
                    shuffle_patches = False,
                    samples_per_volume = len(self.grid_sampler),
                    sampler = self.grid_sampler,
                    num_workers = self.num_workers
                )   

    def must_aggregate_patches(self):
        return self.aggregate_patches
                
    def get_grid_sampler(self):
        return self.grid_sampler
    
    def train_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.trainset, shuffle=False) if self.is_distributed else None
        return DataLoader(dataset=self.trainset, batch_size=self.training['batch_size'], sampler=sampler,num_workers=0, shuffle=False )

    def val_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.valset, shuffle=False) if self.is_distributed else None
        return DataLoader(dataset=self.valset,   batch_size=self.validation['batch_size'], sampler=sampler,num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.testset,  batch_size=self.test_and_predict['batch_size'])
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predset,  batch_size=self.test_and_predict['batch_size'])

    def get_transform(self, data_augm_v=None):
        if data_augm_v == 0:
            intensity = tio.OneOf({
                tio.RandomNoise(std=0.05): 0.5,
                tio.RandomBiasField(): 0.2,
                #tio.RescaleIntensity(out_min_max=0.1): 0.0
                tio.RandomGamma(log_gamma=(-1, 1)): 0.2,
                tio.Compose({
                    tio.RandomNoise(std=0.05),
                    tio.RandomBiasField()
                }): 0.1
            }, p=1.0 )
            transforms = [intensity]
            transform = tio.Compose(transforms)
        elif data_augm_v == 1:
            intensity = tio.OneOf({
                tio.RandomNoise(mean=(0.0,0.0), std=0.07): 0.5,
                RandomRescale(upper_bound_shift=[-0.4,0.1], noise_std=[0,0.04]): 0.5,
            }, p=0.0 )
            spatial = tio.OneOf({
                tio.RandomFlip(axes=(0,1,2), flip_probability = 1/3, include=['img','label']): 1.0,
                tio.RandomElasticDeformation(num_control_points=10,max_displacement=25, include=['img','label']): 0.0,
            }, p=1.0 )
            transforms = [intensity, spatial]
            transform = tio.Compose(transforms)
        elif data_augm_v == 2:
            intensity = tio.OneOf({
                tio.RandomNoise(mean=(0.0,0.0), std=0.08): 0.4,
                RandomRescale(upper_bound_shift=[-0.4,0.2], noise_std=[0,0.04]): 0.4,
                RandomWave(intensity=[0.05,0.15], frequency=[0,30]): 0.2,
            }, p=1.0 )
            spatial = tio.OneOf({
                tio.RandomFlip(axes=(0,1,2), flip_probability = 1/3, include=['img','label']): 0.5,
                tio.RandomElasticDeformation(num_control_points=12,max_displacement=25, include=['img','label']): 0.5,
            }, p=1.0 )
            transforms = [intensity, spatial]
            transform = tio.Compose(transforms)
        elif data_augm_v == 3:
            intensity = tio.OneOf({
                tio.RandomNoise(mean=(0.0,0.0), std=0.08): 0.3,
                RandomRescale(upper_bound_shift=[-0.4,0.2], noise_std=[0,0.04]): 0.3,
                RandomWave(intensity=[0.05,0.15], frequency=[0,30]): 0.2,
                tio.RandomBlur():0.2
            }, p=1.0 )
            spatial = tio.OneOf({
                tio.RandomFlip(axes=(0,1,2), flip_probability = 1/3, include=['img','label']): 0.5,
                tio.RandomElasticDeformation(num_control_points=12,max_displacement=25, include=['img','label']): 0.5,
            }, p=1.0 )
            transforms = [intensity, spatial]
            transform = tio.Compose(transforms)
        elif data_augm_v == 4:
            intensity = tio.OneOf({
                tio.RandomNoise(mean=(0.0,0.0), std=0.08): 0.3,
                RandomRescale(upper_bound_shift=[-0.4,0.2], noise_std=[0,0.04]): 0.3,
                RandomWave(intensity=[0.05,0.15], frequency=[0,30]): 0.2,
                tio.RandomBlur():0.2
            }, p=1.0 )
            spatial = tio.OneOf({
                tio.RandomFlip(axes=(0,1,2), flip_probability = 1/3): 0.5,
                tio.RandomElasticDeformation(num_control_points=12,max_displacement=25): 0.5,
            }, p=1.0 )
            transforms = [intensity, spatial]
            transform = tio.Compose(transforms)   
        elif data_augm_v == 'bones_1':
            intensity = tio.OneOf({
                tio.RandomNoise(mean=(0.0,0.0), std=0.08): 0.5,
                RandomWave(intensity=[0.05,0.05], frequency=[0,30]): 0.25,
                tio.RandomBlur():0.25
            }, p=1.0 )
            spatial = tio.OneOf({
                tio.RandomFlip(axes=(0,1,2), flip_probability = 1/3): 0.5,
                tio.RandomElasticDeformation(num_control_points=12,max_displacement=35): 0.5,
            }, p=1.0 )
            transforms = [intensity, spatial]
            transform = tio.Compose(transforms)   
        elif data_augm_v == 'bones_2':
            intensity = tio.OneOf({
                tio.RandomNoise(mean=0, std=0.1): 0.5,
                tio.RandomBlur(std=6):0.25
            }, p=1.0 )
            spatial = tio.OneOf({
                tio.RandomFlip(axes=(0,1,2), flip_probability = 1/3): 0.5,
                tio.RandomAffine(scales=(0.9,1.1),degrees=(0,20),default_pad_value=0): 0.5,
                #tio.RandomElasticDeformation(num_control_points=12,max_displacement=40): 0.5,
            }, p=1.0 )
            transforms = [intensity, spatial]
            transform = tio.Compose(transforms)   
        elif data_augm_v == 'pad_or_crop':
            transform = tio.CropOrPad(target_shape=self.patch_size)
        elif data_augm_v == 5:
            intensity = tio.OneOf({
                tio.RandomNoise(mean=(0.0,0.0), std=0.08): 0.4,
                RandomRescale(upper_bound_shift=[-0.4,0.2], noise_std=[0,0.04]): 0.4,
                RandomWave(intensity=[0.05,0.15], frequency=[0,30]): 0.2,
            }, p=1.0 )
            spatial = tio.OneOf({
                tio.RandomFlip(axes=(0,1,2), flip_probability = 1/3, include=['img','label']): 0.5
            }, p=1.0 )
            transforms = [intensity, spatial]
            transform = tio.Compose(transforms)
        ## Follow debug options
        elif data_augm_v == -1: #noise
            intensity = tio.OneOf({
                tio.RandomNoise(mean=(0.0,0.0), std=0.10): 0.5,
                tio.RandomNoise(mean=(0.0,0.0), std=0.10): 0.5
            }, p=1.0 )
            transforms = [intensity]
            transform = tio.Compose(transforms)
        elif data_augm_v == -2: #waves
            intensity = tio.OneOf({
                RandomWave(intensity=[0.15,0.15], frequency=[20,30]): 0.5,
                RandomWave(intensity=[0.15,0.15], frequency=[20,30]): 0.5
            }, p=1.0 )
            transforms = [intensity]
            transform = tio.Compose(transforms)
        elif data_augm_v == -3: #uniform
            intensity = tio.OneOf({
                tio.RandomNoise(mean=(0.0,0.0), std=0.00): 0.5,
                tio.RandomNoise(mean=(0.0,0.0), std=0.00): 0.5
            }, p=1.0 )
            transforms = [intensity]
            transform = tio.Compose(transforms)
        else:
            transform = None

        return transform
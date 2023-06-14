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


from collections import defaultdict
from typing import Tuple

import torch
from typing import Tuple, Union, Dict, Sequence
from torchio.data.subject import Subject
from torchio.transforms import IntensityTransform
from torchio.transforms.augmentation import RandomTransform


class RandomRescale(RandomTransform, IntensityTransform):
    def __init__(
            self,
            upper_bound_shift: Union[float, Tuple[float, float]] = (-0.3, 0.3),
            noise_std: Union[float, Tuple[float, float]] = 0.0,
            **kwargs
            ):
        """
        This function execute a random intensity shift of the values of a volume. 
        
        Parameters
        ----------
        upper_bound_shift: 
            shift of the upper bound
        """
        super().__init__(**kwargs)
        self.upper_bound_shift = self._parse_range(upper_bound_shift, 'upper_bound_shift')
        self.noise_std         = self._parse_range(noise_std, 'noise_std', min_constraint=0)

    def apply_transform(self, subject: Subject) -> Subject:
        shift, noise_std, seed = self.get_params(self.upper_bound_shift, self.noise_std)
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            arguments['shift'][name]           = shift
            arguments['noise_std'][name]       = noise_std
            arguments['seed'][name]            = seed
        transform = Rescale(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, 
            upper_bound_shift: Tuple[float, float],
            noise_std: Tuple[float, float],
            ) -> Tuple[float, float, int]:
        shift     = self.sample_uniform(*upper_bound_shift)#.item()
        noise_std = self.sample_uniform(*noise_std)#.item()
        seed      = self._get_random_seed()
        return shift, noise_std, seed



class Rescale(IntensityTransform):
    def __init__( self, 
            shift: Union[float, Dict[str, float]],
            noise_std: Union[float, Dict[str, float]],
            seed: Union[int, Sequence[int]],
            **kwargs
            ):
        super().__init__(**kwargs)
        self.shift = shift
        self.noise_std = noise_std
        self.seed = seed
        self.args_names = 'shift','noise_std','seed'
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        shift, noise_std, seed = args = self.shift, self.noise_std, self.seed
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                shift, noise_std, seed = (arg[name] for arg in args)
            with self._use_seed(seed):
                noisy_image = rescale(image.data, shift, noise_std)
            image.set_data(noisy_image)
        return subject


def rescale(tensor, shift, noise_std):
    #min = torch.min(tensor) 
    #max = torch.max(tensor) 
    tensor = tensor *(1+shift) +torch.randn(tensor.shape)*noise_std
    return torch.clip(tensor, min=0.0)






class RandomWave(RandomTransform, IntensityTransform):
    def __init__(
            self,
            intensity: Union[float, Tuple[float, float]] = (-0.3, 0.3),
            frequency: Union[float, Tuple[float, float]] = 0.0,
            **kwargs
            ):
        """
        This function execute a random intensity shift of the values of a volume. 
        
        Parameters
        ----------
        intensity: float, tuple of floats
            maximal shift due to the wave. It can also be a range of intensities between which a value is sampled
        frequency: float, tuple of floats
            maximal frequency or range of frequencies
        """
        super().__init__(**kwargs)
        self.intensity = self._parse_range(intensity, 'intensity', min_constraint=0)
        self.frequency = self._parse_range(frequency, 'frequency', min_constraint=0)

    def apply_transform(self, subject: Subject) -> Subject:
        intensity, frequency, seed = self.get_params(self.intensity, self.frequency)
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            arguments['intensity'][name]  = intensity
            arguments['frequency'][name]  = frequency
            arguments['seed'][name]       = seed
        transform = Ripple(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, 
            intensity: Tuple[float, float],
            frequency: Tuple[float, float],
            ) -> Tuple[float, float, int]:
        intensity     = self.sample_uniform(*intensity)#.item()
        frequency = self.sample_uniform(*frequency)#.item()
        seed      = self._get_random_seed()
        return intensity, frequency, seed



class Ripple(IntensityTransform):
    def __init__( self, 
            intensity: Union[float, Dict[str, float]],
            frequency: Union[float, Dict[str, float]],
            seed: Union[int, Sequence[int]],
            **kwargs
            ):
        super().__init__(**kwargs)
        self.intensity = intensity
        self.frequency = frequency
        self.seed = seed
        self.args_names = 'intensity','frequency','seed'
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        intensity, frequency, seed = args = self.intensity, self.frequency, self.seed
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                intensity, frequency, seed = (arg[name] for arg in args)
            with self._use_seed(seed):
                noisy_image = ripple(image.data, intensity, frequency)
            image.set_data(noisy_image)
        return subject


def ripple(tensor, intensity, frequency):
    '''
    frequency is meant to be how many cycles of a wave there are within the dataset in a random direction of propagation
    '''
    ripples = torch.zeros(tensor.shape[1:])
    ripple_ver = torch.FloatTensor(3).uniform_(0,1)
    ripple_ver = ripple_ver/torch.linalg.norm(ripple_ver+0.001)
    ripple_ver = torch.floor(ripple_ver*frequency)
    idx = ripple_ver.type(torch.IntTensor).cpu().detach().numpy()
    for i, value in enumerate(ripple_ver):
        if value == 0:
            idx[i]=1
    ripples[idx[0],idx[1],idx[2]] = intensity
    torch.fft.irfftn(ripples, norm='forward', s=ripples.shape, out=ripples)/2
    tensor_ripples = ripples.repeat(tensor.shape[0], 1,1,1)
    tensor = tensor *( 1+ tensor_ripples)
    return torch.clip(tensor, min=0.0)
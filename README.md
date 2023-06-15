# Neural network framework for 3D data
The aim of this framework is to give researchers an integrated system for applying 3D networks on 3D data, with a patch-based approach. The use of a patch-based approach allows to reduce the memory requirements of the network, and to apply the network on data of arbitrary size. The framework is supposed to be easily configurable from configuration files that store information about where is the training/validation data, network architecture, the loss function, the data augmentation techniques, etc. 
A configuration file is a YAML file that contains all the information needed to run the framework for three purposes: training/validation, testing and inference. 


## General information
The framework is based on the [PyTorch](https://pytorch.org/) library, the [torchio](https://torchio.readthedocs.io/) library, [Pytorch Lightning](https://www.pytorchlightning.ai/), [wandb](https://wandb.ai/) for logging and it is driven by configuration files (YAML files).
The framework has several network architectures already implemented, but it is easy to add new ones. The framework is designed to be easily extensible, and to allow the user to add new networks, new loss functions, new metrics, new data augmentation techniques, etc. Although the networks have been used for anomaly detection, they can be extended to other tasks.

If you use this framework and you have liked using it, please star it :star2: and cite the following paper :page_facing_up: for reference, as it the basic framework of the research work presented in:

D. Iuso, S. Chatterjee, S. Cornelissen, D. Verhees, J. De Beenhouwer, J. Sijbers. "Voxel-wise classification for porosity investigation of additive manufactured parts with 3D unsupervised and (deeply) supervised neural networks" ( https://doi.org/10.48550/arXiv.2305.07894 )

## Requirements
- pytorch
- Pytorch Lightning
- torchio
- wandb
- yaml
- Some free time to adapt the framework to your needs

## License
The software is released under the GNU General Public License v3.0 - See the LICENSE file for more details.

## Example of use
The software is meant to be used from the command line by calling one of the three scripts provided:

- train.py
- test.py
- predict.py

The scripts accept many optional arguments and one mandatory argument, which is the path to the configuration file:
```
python train.py --config_file path/to/config/file
```
As could be seen from the scripts, the content of the configuration file is divided into sections, each one with a specific purpose. Examples of configuration files are provided in the "examples" folder. Most of the times, the content of (each section of) the configuration file is directly supplied to a module (e.g. a network model, the pytorch-lightning Trainer, the logger, etc.), which prevents the user to write (and to-be-kept-updated) boilerplate code. 


## Implementation details
The framework is composed of several modules, each one with a specific purpose and supposed to not be dependent on the others. Some more words on [Network architecture](#network-architectures) and [Data loaders and data augmentation](#data-loaders-and-data-augmentation) will be given in the following sections.


### Network architectures
As the original implementation of all the implemented networks were 2D, we directly adapted them to 3D. The implemented base network architectures are:

- UNet
- UNet++
- UNet 3+
- MSS-UNet
- VAE
- ceVAE
- gmVAE
- vqVAE

The architecture is "scalable" as it is possible to add new convolutional blocks, usually by changing only one parameter on network initialisation (or by changing/adding a parameter in the configuration file).


### Data loaders and data augmentation
The framework uses the torchio library for data loading and data augmentation. The framework is designed to be easily extensible, and to allow the user to add new data augmentation techniques. The provided dataloader is designed to read in each folder contained in the training/validation dataset a stack of 2D tif files that are the slices of a 3D volume (associated to one subject/sample). You may want to declare a new dataloader based on the exemplary one provided.

### Supplementary (incomplete) features
Some of the provided network models supported [deepspeed](https://github.com/microsoft/DeepSpeed) for a while, but the support got lost after pl-lightning latest changes. I used the deepspeed library for having big networks (and optimisers, optimiser states, gradients) split among GPUs. This functionality should be restored soon.


### Bug and issues
If you find any bug or issue, please open an issue on the github page of the project. I will try to address it as soon as possible, but I cannot guarantee any time frame. Any contribution (i.e. pull request) is (very) welcome.

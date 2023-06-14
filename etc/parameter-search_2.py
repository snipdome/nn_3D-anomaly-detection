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
import os, json, yaml
import argparse, pathlib
import multiprocessing as mp
import subprocess
import nn.utils.utils as utils
from termcolor import colored

def fun(fun_path, *args):
    subprocess.run( [fun_path, *list(args)], env=os.environ.copy(), check=True )

def main(args, args_for_subp):
    
    exes = {}
    if args['train']:
        exes['train'] = os.environ['AT'] + '/Projects/nn/train.py'
    if  args['test']:
        exes['test'] = os.environ['AT']  + '/Projects/nn/test.py'
    if  args['predict']:
        exes['predict'] = os.environ['AT']  + '/Projects/nn/predict.py'
    
    config_file = args['config_file']
        
    #
    # Parameter search
    #

    gamma = np.concatenate((np.linspace(0, 2, 7), np.linspace(0, 2, 5)))
    gamma = np.sort(gamma)
    gamma = np.unique(gamma)
    gamma = gamma[1:] # remove the zero
    gamma = [round(x,3) for x in gamma]
    
    total_param_n = np.prod(len(gamma)) # total number of elements in the parameter space
    
    gamma_string = []
    for i in range(total_param_n):
        gamma_string.append(str(gamma[i]))
        gamma_string[-1] = 'model.loss.parameters.gamma=' + gamma_string[-1]
        
    if args['n'] is not None:
        args['start_n'] = int(args['n'])
        args['end_n']   = int(args['n'])+1
    else:
        args['start_n'] = int(args.get('start_n'), 0) if (args.get('start_n') is not None) else 0
        args['end_n']   = int(args.get('end_n', total_param_n)) if (args.get('end_n') is not None) else total_param_n
    
    #
    # K-fold validation
    #
    total_k_space = 5 # k-fold space
    base_path = str(pathlib.Path(config_file).parent.absolute()) + '/dataset-links/'
    pred_basepath = os.path.join(base_path,'predictions')
    train_paths       = []
    train_labels      = []
    train_likelihoods = []
    valid_paths       = []
    valid_labels      = []
    valid_likelihoods = []
    test_paths        = []
    test_labels       = []
    for k in range(total_k_space):
        train_paths.append(os.path.join(base_path, str(k), 'trainset','image'))
        train_paths[-1] = 'dataloader.training.dataset.path=' + train_paths[-1]
        train_labels.append(os.path.join(base_path, str(k), 'trainset','object-label'))
        train_labels[-1] = 'dataloader.training.dataset.label=' + train_labels[-1]
        train_likelihoods.append(os.path.join(base_path, str(k), 'trainset','pore-label'))
        train_likelihoods[-1] = 'dataloader.training.dataset.sampling_likelihood=' + train_likelihoods[-1]
        valid_paths.append(os.path.join(base_path, str(k), 'valset','image'))
        valid_paths[-1] = 'dataloader.validation.dataset.path=' + valid_paths[-1]
        valid_labels.append(os.path.join(base_path, str(k), 'valset','object-label'))
        valid_labels[-1] = 'dataloader.validation.dataset.label=' + valid_labels[-1]
        valid_likelihoods.append(os.path.join(base_path, str(k), 'valset','pore-label'))
        valid_likelihoods[-1] = 'dataloader.validation.dataset.sampling_likelihood=' + valid_likelihoods[-1]
        test_paths.append(os.path.join(base_path, str(k), 'valset','image'))
        test_paths[-1] = 'dataloader.test_and_predict.dataset.path=' + test_paths[-1]
        test_labels.append(os.path.join(base_path, str(k), 'valset','pore-label'))
        test_labels[-1] = 'dataloader.test_and_predict.dataset.label=' + test_labels[-1]

    if args['k'] is not None:
        args['start_k'] = int(args['k'])
        args['end_k']   = int(args['k'])+1
    else:
        args['start_k'] = int(args.get('start_k', 0)) if (args.get('start_k') is not None) else 0
        args['end_k']   = int(args.get('end_k',   total_k_space)) if (args.get('end_k') is not None) else total_k_space
        
    
    for k in range(args['start_k'], args['end_k']):
        for i in range(args['start_n'], args['end_n']):

            name = 'net-ps_g-' +str(gamma[i]) +'_k-' +str(k) 
            model_name = 'model.name=' + name

            
            for modality in exes.keys():

                if modality == 'train':
                    parameters_to_change = [model_name, gamma_string[i], 
                        train_paths[k], train_labels[k], train_likelihoods[k], valid_paths[k], valid_labels[k], valid_likelihoods[k]]
                elif modality == 'test':
                    parameters_to_change = [model_name, gamma_string[i], 
                        test_paths[k], test_labels[k], 'project=Parameter-search-test']
                elif modality == 'predict':
                    pred_results = 'dataloader.test_and_predict.dataset.results=' + os.path.join(pred_basepath, name)
                    parameters_to_change = [model_name, gamma_string[i], 
                        test_paths[k], pred_results]

                print('parameter-search: Starting job n {}:  '.format(i) + colored(name,'red'))

                if args['dry_run']:
                    print(exes[modality], '--config_file', config_file, *args_for_subp, '--set', *parameters_to_change)
                else:
                    fun(exes[modality], '--config_file',config_file, *args_for_subp, '--set', *parameters_to_change)
        

if __name__ == '__main__':

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--config_file', required=False, default=None)
    parent_parser.add_argument('--dry_run',     required=False, default=False, action=argparse.BooleanOptionalAction)
    parent_parser.add_argument('--start_n',     required=False)
    parent_parser.add_argument('--n',           required=False, default=None)
    parent_parser.add_argument('--end_n',       required=False)
    parent_parser.add_argument('--start_k',     required=False)
    parent_parser.add_argument('--k',           required=False, default=None)
    parent_parser.add_argument('--end_k',       required=False)
    parent_parser.add_argument('--train',   action='store_true')
    parent_parser.add_argument('--test',    action='store_true')
    parent_parser.add_argument('--predict', action='store_true')
    args, args_for_subp = parent_parser.parse_known_args()
    
    args = vars(args)

    main( args, args_for_subp )
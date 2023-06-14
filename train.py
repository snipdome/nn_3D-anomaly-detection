#!/usr/bin/env python
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


import os, json, yaml
from utils import no_torch_utils
from argparse import ArgumentParser

os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = no_torch_utils.find_free_gpus(tolerance=0.2)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--config_file', required=True)
    parent_parser.add_argument('--skip_gpus', '--skip_gpu', '--avoid_gpus', '--avoid_gpu',required=False, default=None, type=no_torch_utils.list_str_to_int)
    parent_parser.add_argument('--use_gpus', '--use_gpu', required=False, default=None, type=no_torch_utils.list_str_to_int)
    parent_parser.add_argument("--set",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="Set a number of key-value pairs "
                             "(do not put spaces before or after the = sign). "
                             "If a value contains spaces, you should define "
                             "it with double quotes: "
                             'foo="this is a sentence". Note that '
                             "values are always treated as strings.")
    args, unknown = parent_parser.parse_known_args()
    
    no_torch_utils.set_gpu_flags(skip_gpus=args.skip_gpus, use_gpus=args.use_gpus)
    
    
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, loggers, seed_everything
from lightning_fabric.utilities.seed import reset_seed
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.plugins import DDPPlugin
import models 
import dataloaders
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
    
torch.set_num_threads(8)                                                                                            

def main(args):
    
    with open(args.config_file) as f:
        cfg = yaml.full_load(f)

    seed_everything(cfg['seed'], workers=True)       
     
    #
    # overwrite values in config-file
    #
    overwrite_cfg_values = no_torch_utils.parse_string_dict_to_structured_dict(no_torch_utils.parse_vars(args.set))
    no_torch_utils.update_dict(cfg,overwrite_cfg_values)
    
    cfg['version'] = 'v'+str(cfg.get('version')) if cfg.get('version') is not None else '0'
    
    #
    # Model
    #
    data_module = eval("dataloaders."+cfg['dataloader']['name']+'(**cfg[\'dataloader\'])')
    model = eval(cfg['model']['type']+'(**cfg[\'model\'])')
    model.set_dataloader(data_module)
    
    resume=cfg["training"].get('checkpoint','allow')
    os.makedirs(cfg["model"]["checkpoint_path"], exist_ok=True)
    model = torch.compile(model) if cfg["model"].get('compiled', False) else model
    
    
    #
    # Loggers
    #
    logs = []
    if cfg.get('wandb') is not None:
        if cfg['wandb']['args'].get('save_dir') is None:
            cfg['wandb']['args']['save_dir'] = cfg['model']['log_path']
        os.makedirs(cfg['wandb']['args']['save_dir'], exist_ok=True)
        wandb_logger = WandbLogger(project=cfg['project'], name=cfg['model']['name'], id=cfg['model']['name'] +"-" +cfg['version'], **cfg['wandb']['args']) 
        if cfg['wandb'].get('watch', {}) is not None:
            wandb_logger.watch(model, log_freq=cfg['dataloader']['training']['queue_length']//cfg['dataloader']['training']['batch_size'], **cfg['wandb']['watch'])
            logs.append(wandb_logger)
    tb_logger = loggers.TensorBoardLogger(save_dir=cfg["model"]["log_path"], name=cfg['model']['name'], version=cfg['version'], default_hp_metric=False,)
    logs.append(tb_logger)
    
    
    ###
    callbacks = []
    checkpoint_callback = ModelCheckpoint( 
        dirpath=cfg['model']["checkpoint_path"], 
        save_top_k=2,
        monitor="Train/loss",
        save_last=True,
        filename=cfg['model']['name']+'-{epoch}-'+cfg['version']
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = cfg['model']['name']+"-{epoch}-last-"+cfg['version']
    callbacks.append(checkpoint_callback)
    
    if cfg['training'].get('callbacks',{}) is not None:
        if cfg['training']['callbacks'].get('early_stopping', None):
            stop_callback = EarlyStopping(
                **cfg['training']['callbacks']['early_stopping'])
            callbacks.append(stop_callback)
        
        
    if cfg['training'].get('strategy')=='ddp':
        cfg['training']['args'].update({'strategy':DDPPlugin(find_unused_parameters=False)})
    
    
    trainer = Trainer( 
        callbacks=callbacks,
        logger=logs,
        **cfg['training']['args'],
    )
    
    reset_seed()

    
    if cfg['training'].get('checkpoint',None) == 'last':
        print('Checkpoint path has not been set. It will be inferred...')
        cfg['training']['checkpoint'] = no_torch_utils.find_last_checkpoint(cfg['model']['checkpoint_path'], cfg['model']['name'], cfg['version'])

    if cfg['model'].get('sanitize_checkpoint',False):
        print("Requested sanitizing checkpoint...")
        model = model.load_from_checkpoint(cfg['training']['checkpoint'])
        with torch.no_grad():
            for p in model.parameters():
                if (~torch.isfinite(p.data)).any():
                    print('Invalid values found in parameter: ', p)
                    p.data[~torch.isfinite(p.data)] = 0
        trainer.fit(model=model, datamodule=data_module)
    else:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg['training'].get('checkpoint'))


#os.environ["NCCL_DEBUG_FILE"]="/data/home/diuso/tmp/logs"
os.environ['NCCL_DEBUG']='INFO'
#os.environ['NCCL_DEBUG_SUBSYS']='ALL'
'''
os.environ['NCCL_IB_DISABLE']='1'
os.environ['NCCL_SOCKET_IFNAME']='lo'
os.environ['NCCL_P2P_DISABLE']='1' #it has to do with intel
os.environ['MASTER_ADDR']='localhost'
os.environ['MASTER_PORT']='19939'
torch.distributed.init_process_group(backend='nccl', init_method='env://')
'''


if __name__ == '__main__':
    main( args )
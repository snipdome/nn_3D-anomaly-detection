#!/usr/bin/env python
import os, json, yaml, glob
from utils import no_torch_utils
from argparse import ArgumentParser

os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = no_torch_utils.find_free_gpus(tolerance=0.05)


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
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import ModelSummary
import models 
import dataloaders
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
#import pl_bolts.models as pl_bolts

torch.set_num_threads(20)            


def main(args):
    
    with open(args.config_file) as f:
        cfg = yaml.full_load(f)
        
    #
    # overwrite values in config-file
    #
    overwrite_cfg_values = no_torch_utils.parse_string_dict_to_structured_dict(no_torch_utils.parse_vars(args.set))
    no_torch_utils.update_dict(cfg,overwrite_cfg_values)
    
    cfg['version'] = 'v'+str(cfg.get('version')) if cfg.get('version') is not None else '0'

    #
    # Model
    #
    model = eval(cfg['model']['type']+'(**cfg[\'model\'])')
    
    data_module = eval("dataloaders."+cfg['dataloader']['name']+'(**cfg[\'dataloader\'])')

    model.set_dataloader(data_module)

    #if cfg['test_and_predict']['arguments'].get('gpus') is None:
    #    cfg['test_and_predict']['arguments']['gpus'] = find_free_gpu([4,5,6,7,3,2,1,0])
        
    trainer = Trainer(
        **cfg['test_and_predict']['args'],
        logger=False
    )

    
    if cfg['test_and_predict'].get('checkpoint',None) == 'last':
        print('Checkpoint path has not been set. It will be inferred...')
        cfg['test_and_predict']['checkpoint'] = no_torch_utils.find_last_checkpoint(cfg['model']['checkpoint_path'], cfg['model']['name'], cfg['version'])
        
        
    trainer.predict(model=model, datamodule=data_module, ckpt_path=cfg['test_and_predict']['checkpoint'])

if __name__ == '__main__':
    main( args )
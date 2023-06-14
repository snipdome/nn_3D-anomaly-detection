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


import wandb, torch, numpy as np
    
def log_wandb_image(logger, name, y):
    #if self.log_wandb_images:
    #    if batch_nb == 0 and self.current_epoch!=0:
    if len(y.shape[2:]) == 3:
        logger.log_image(name, [convert_to_wandb_image(torch.clamp(y[0,0,:,:,y.shape[4]//2],0,1))])
    else:
        logger.log_image(name, [convert_to_wandb_image(torch.clamp(y[0,0,:,:],0,1))])
        
        
def convert_to_wandb_image(image):
    '''
    Converts a float image in 0-1 to a wandb.Image in 0-255
    '''
    return wandb.Image((255*image.detach().cpu().numpy()).astype(np.uint8))
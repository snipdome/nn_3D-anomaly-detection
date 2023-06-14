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


from tensorboard.backend.event_processing import event_multiplexer
import os, argparse, glob, ast
from packaging import version
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
from scipy import ndimage
import numpy as np


def find_path_folders_inside_directory(base_path, common_name='*', want_to_get_last=None):
    list_of_folders = []
    for file in glob.glob( os.path.join(base_path, common_name) ):
        if os.path.isdir(file):
            list_of_folders.append(file)
            
    #if (want_to_get_last is not None) and (len(list_of_folders)>1):
    #    list_of_folders = [sorted(list_of_folders)[-1]] if want_to_get_last else [sorted(list_of_folders)[1]]
    return list_of_folders

def convertEventMultiplexer_to_Dataframe(em):
    
    # Print tags of contained entities, use these names to retrieve entities as below
    #print(em.Runs())
    scalars = {runName : data['scalars'] for (runName, data) in em.Runs().items() if len(data['scalars'])>0}
    data = {}
    for runName, runtags in scalars.items():
        data[runName] = {}
        for tag in runtags:
            #em.Scalars returns ScalarEvents array holding wall_time, step, value per time step (shape series_length x 3)
            #print(mpl.Scalars(runName, tag)[0])
            run_scalars = [(s.step, s.value) for s in em.Scalars(runName, tag)]
            data[runName][tag] = run_scalars

            
    for runName, runPath in em.RunPaths().items():
        data[runName]['runPath'] = runPath#['hparams'] = (alpha, beta, k)
            
    return pd.DataFrame.from_dict(data)

def add_metadata_info(df):
    for index in df.index.values:
        if index == 'runPath': 
            for runName in df.columns.values:
                extracted_hparams = df.loc[index,runName].split('/')[-2].split('_') # assumes that the name of the network is name_param1_param2_param3
                if extracted_hparams[1].split('-')[0] == 'a': #then it is the right parameter search
                    df.at['alpha', runName] = float(extracted_hparams[1].split('-')[1])
                    df.at['beta', runName]  = float(extracted_hparams[2].split('-')[1])
                    df.at['k', runName]     = int(extracted_hparams[3].split('-')[1])
                else:
                    df.drop(runName, axis=1, inplace=True)
    return df
                
    

def main(args):


    ''' 
        Read tensorboard logs from folders
    '''
    list_of_experiments = find_path_folders_inside_directory( base_path=args.path_to_runs, common_name=args.common_name_of_runs )
    list_of_runs        = []
    for experiment_path in list_of_experiments:
        for run in find_path_folders_inside_directory(base_path=experiment_path, common_name='v*', want_to_get_last=args.want_to_get_last):
            list_of_runs.append( run )

    list_of_runs_as_dict = {'run'+str(i): list_of_runs[i] for i in range(0, len(list_of_runs))}
    em = event_multiplexer.EventMultiplexer(list_of_runs_as_dict)
    em.Reload()
    
    '''
        Convert data to pandas dataframe
    '''
    df = convertEventMultiplexer_to_Dataframe(em)
      
      
    '''
        Select useful information
    '''     
    df = add_metadata_info(df)
    alpha_list = []
    beta_list  = []
    k_list     = []
    
    for runName in df.columns.values:
        alpha_list.append(df.loc['alpha',runName]) if df.loc['alpha',runName] not in alpha_list else None
        beta_list.append(df.loc['beta',runName])   if df.loc['beta',runName] not in beta_list   else None
        k_list.append(df.loc['k',runName])         if df.loc['k',runName] not in k_list         else None
    alpha_list.sort()
    beta_list.sort()
    k_list.sort()
    alpha = [x for x in alpha_list]
    beta  = [x for x in beta_list]
    k     = [x for x in k_list]
    zs = np.zeros((len(alpha),len(beta), 5),dtype=np.float32)
    #zs = zs + np.random.randn(*zs.shape)*0.0000001
        
    for runName in df.columns.values:
        current_alpha = alpha.index(df.loc['alpha',runName])
        current_beta  = beta.index(df.loc['beta',runName])
        current_k     = k.index(df.loc['k',runName])
        try:
            value = df.loc[args.metric,runName][-1][-1] # takes the last one
        except:
            value = 0
        if current_k==0:
            print('({},{},{}) {}'.format(current_alpha,current_beta,current_k,value))
        if float(value) > zs[current_alpha,current_beta,current_k]:
            zs[current_alpha,current_beta,current_k] = float(value)

    os.makedirs('etc/out', exist_ok=True)

    sns.set_theme(style="white")
    #fig = plt.figure(1, figsize=(14, 6), dpi=80) larger
    fig = plt.figure(1, figsize=(10, 4), dpi=80)
    # Create axes for each plot. If the number of plots is rectangular (3x3, 3x5, 4x2), then the if/else statement is not needed
    axes = list()
    i_turn = 3
    for i in range(5):
        if i<i_turn:
            axes.append( plt.subplot2grid(shape=(2,6), loc=(0,2*i), colspan=2) )
        else:
            axes.append( plt.subplot2grid(shape=(2,6), loc=(1,1+2*(i-i_turn)), colspan=2) )

    sns.set(context="notebook", style="whitegrid",  rc={"axes.axisbelow": False})
    

    plt.rc('font', size=18)
    plt.rc('axes', titlesize=20) #fontsize of the title
    plt.rc('axes', labelsize=20) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=18) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=18) #fontsize of the y tick labels
    df_list=[]
    # Rotate the starting point around the cubehelix hue circle
    for i, (ax, s) in enumerate(zip(axes, np.linspace(0, 3, len(axes)+1))):
        
        ax.set_title('Dice-Sørensen score')
        
        # Create a cubehelix colormap to use with kdeplot
        #cmap = sns.cubehelix_palette(start=s, light=0.8, as_cmap=True)
        cmap = sns.cubehelix_palette(start=0, dark=0.15, light=.6, reverse=False, as_cmap=True)

        #zs = alpha_grid+beta_grid+np.random.rand(4,4)
        #df = pd.DataFrame(zs[:,:,0], columns=alpha, index=beta)
        df_list.append( pd.DataFrame(zs[:,:,i], columns=alpha, index=beta) )
        
        #ndimage.zoom creates a refined grid which helps to obtain much smoother contour lines.
        smooth_scale = 3
        z = ndimage.zoom(df_list[i].to_numpy(), smooth_scale)
        
        alpha_smooth = np.linspace(alpha[0], alpha[-1], len(alpha)*smooth_scale)
        beta_smooth  = np.linspace( beta[0],  beta[-1], len(beta)*smooth_scale)
        
        #cntr = ax.contourf(alpha_smooth,beta_smooth, z, levels=np.linspace(z.min(),z.max(),6), cmap=cmap)
        vmin = zs[:,:,i].min() if (zs[:,:,i].min()<0.5) else 0.5
        ax = sns.heatmap(df_list[i], annot=True, alpha=1, cmap=cmap, cbar=False, ax=ax, fmt='.3f', vmin=0.5)
        ax.invert_yaxis()
        ax.set( xlabel = "beta", ylabel = "alpha")
        ax.tick_params(axis='x', rotation=0)
        if True:
            figtosave=plt.figure('tosave'+str(i), figsize=(4.3, 2.7), dpi=80)
            #axtosave = sns.heatmap(df_list[i], annot=True, alpha=1, cmap=cmap, cbar=False, vmin=vmin, vmax=zs[:,:,i].max())
            axtosave = sns.heatmap(df_list[i], annot=True, alpha=1, cmap=cmap, cbar=False, fmt='.3f', vmin=zs[:,:,i].min()*0.9)
            axtosave.invert_yaxis()
            axtosave.set_title('Dice-Sørensen - Fold '+str(i+1))
            axtosave.set( xlabel = "beta", ylabel = "alpha")
            axtosave.tick_params(axis='x', rotation=0)
            #axtosave.tick_params(axis='both', which='major', labelsize=16)
            #plt.show()
            figtosave.savefig('etc/out/alpha-beta-'+str(i),bbox_inches='tight')
            plt.figure(1)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.8)
    ''' 
    if True:
        for i, (ax, s) in enumerate(zip(axes, np.linspace(0, 3, len(axes)+1))):
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig('etc/out/graph_'+str(i),bbox_inches=extent.expanded(1.3,1.2)) '''

    plt.rc('font', size=18)
    plt.rc('axes', titlesize=20) #fontsize of the title
    plt.rc('axes', labelsize=20) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=18) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=18) #fontsize of the y tick labels
    plt.figure(2, figsize=(5, 2.4), dpi=80)
    axes_avg = list()
    axes_avg.append( plt.subplot2grid(shape=(1,2), loc=(0,0)) )
    axes_avg[-1].set_title('Average Dice-Sørensen')
    axes_avg.append( plt.subplot2grid(shape=(1,2), loc=(0,1)) )
    axes_avg[-1].set_title('Dice-Sørensen st.err.')
    '''
    for k in range(zs.shape[2]):
        zs[:,:,k] -= np.mean(zs[:,:,k])
        zs[:,:,k] /= np.std(zs[:,:,k])
        #zs[:,:,k] = zs[:,:,k]-np.min(zs[:,:,k])
        #zs[:,:,k] /= np.max(zs[:,:,k])
    '''

    # Rotate the starting point around the cubehelix hue circle
    for i, (ax, s) in enumerate(zip(axes_avg, np.linspace(0, 3, len(axes_avg)+1))):
        #cmap = sns.cubehelix_palette(start=s, light=0.8, as_cmap=True)
        cmap = sns.cubehelix_palette(start=0, dark=0.2, light=.55, reverse=False, as_cmap=True)
        ax.set_box_aspect(1)
        if i == 0:
            z = np.mean(zs,axis=2)
            avg_df = pd.DataFrame(z, columns=alpha, index=beta)
            #avg_df = avg_df.round(3)
            #z1 = ndimage.zoom(avg_df.to_numpy(), smooth_scale)
            #cntr = ax.contourf(alpha_smooth,beta_smooth, z, levels=np.linspace(z.min(),z.max(),10), cmap=cmap)
            vmin = z.min() if (z.min()<0.5) else 0.6
            ax = sns.heatmap(avg_df, annot=True, alpha=1, cmap=cmap, cbar=False, ax=ax, vmin=vmin, fmt='.3f')
        else:
            zs_v = np.std(zs,axis=2) / np.sqrt(5)
            std_df = pd.DataFrame(zs_v, columns=alpha, index=beta)
            #std_df = std_df.round(3)
            z2 = ndimage.zoom(std_df.to_numpy(), smooth_scale)
            #cntr = ax.contourf(alpha_smooth,beta_smooth, z, levels=np.linspace(z.min(),z.max(),10), cmap=cmap)
            ax = sns.heatmap(std_df, annot=True, alpha=1, cmap=cmap, cbar=False, ax=ax, fmt='.3f')
        ax.invert_yaxis()
        ax.set( xlabel = "beta", ylabel = "alpha")
        if True:
            figtosave=plt.figure('avg_tosave'+str(i), figsize=(4.3, 2.7), dpi=80)
            if i==0:
                #axtosave = sns.heatmap(avg_df, annot=True, alpha=1, cmap=cmap, cbar=False, vmin=vmin)
                axtosave = sns.heatmap(avg_df, annot=True, alpha=1, cmap=cmap, cbar=False, fmt='.3f')
                axtosave.invert_yaxis()
                axtosave.set_title('Average Dice-Sørensen')
                axtosave.set( xlabel = "beta", ylabel = "alpha")
                axtosave.tick_params(axis='x', rotation=0)
                figtosave.savefig('etc/out/average-alpha-beta',bbox_inches='tight')
            else:
                #axtosave = sns.heatmap(std_df, annot=True, alpha=1, cmap=cmap, cbar=False)
                axtosave = sns.heatmap(std_df, annot=True, alpha=1, cmap=cmap, cbar=False, fmt='.3f')
                axtosave.invert_yaxis()
                axtosave.set_title('Dice-Sørensen st.err.')
                axtosave.set( xlabel = "beta", ylabel = "alpha")
                axtosave.tick_params(axis='x', rotation=0)
                figtosave.savefig('etc/out/var-alpha-beta',bbox_inches='tight')
            plt.figure(2)
        

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--path_to_runs', required=True)
    parent_parser.add_argument('--common_name_of_runs', required=False, default='*')
    parent_parser.add_argument('--metric', required=False, default='Test/torch_dice')
    parent_parser.add_argument('--want_to_get_last', required=False, default=True, action=argparse.BooleanOptionalAction)
    args, unknown = parent_parser.parse_known_args()

    main( args )
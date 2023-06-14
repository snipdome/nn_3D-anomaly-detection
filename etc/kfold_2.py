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
        if runName in scalars.keys():
            data[runName]['runPath'] = runPath#['hparams'] = (alpha, beta, k)
            
    return pd.DataFrame.from_dict(data)

def add_metadata_info(df):
    for index in df.index.values:
        if index == 'runPath': 
            for runName in df.columns.values:
                extracted_hparams = df.loc[index,runName].split('/')[-2].split('_') # assumes that the name of the network is name_param1_param2_param3
                if extracted_hparams[1].split('-')[0] == 'g': #then it is the right parameter search
                    df.at['gamma', runName] = float(extracted_hparams[1].split('-')[1])
                    df.at['k', runName]     = int(extracted_hparams[2].split('-')[1])
                else:
                    print('dropped: ' +runName)
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
    gamma_list = []
    k_list     = []
    
    for runName in df.columns.values:
        gamma_list.append(df.loc['gamma',runName]) if df.loc['gamma',runName] not in gamma_list else None
        k_list.append(df.loc['k',runName])         if df.loc['k',runName] not in k_list         else None
    gamma_list.sort()
    k_list.sort()
    gamma = [x for x in gamma_list]
    k     = [x for x in k_list]
    zs = np.zeros((5,len(gamma)),dtype=np.float32)
    #zs = zs + np.random.randn(*zs.shape)*0.0000001
        
    for runName in df.columns.values:
        current_gamma = gamma.index(df.loc['gamma',runName])
        current_k     = k.index(df.loc['k',runName])
        try:
            value = df.loc[args.metric,runName][-1][-1] # takes the last one
        except:
            value = 0
        if float(value) > zs[current_k,current_gamma]:
            print('({},{}) {}'.format(current_k,current_gamma,value))
            zs[current_k,current_gamma] = float(value)



    fig = plt.figure()
    sns.set_theme(style="white")
    sns.set(context="notebook", style="whitegrid",  rc={"axes.axisbelow": False})
    ax = list()
    ax.append( plt.subplot2grid(shape=(1,4), loc=(0,0), colspan=2) )
    ax.append( plt.subplot2grid(shape=(1,4), loc=(0,2)) )
    ax.append( plt.subplot2grid(shape=(1,4), loc=(0,3)) )
    #ax.set_box_aspect(1)

    plt.rc('font', size=18)
    plt.rc('axes', titlesize=20) #fontsize of the title
    plt.rc('axes', labelsize=20) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=18) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=18) #fontsize of the y tick labels

    ax[0].set_title('Dice-Sørensen score')
    #cmap = sns.cubehelix_palette(start=0, light=0.8, as_cmap=True)
    cmap = sns.cubehelix_palette(start=0, dark=0.1, light=.6, reverse=False, as_cmap=True)
    new_k = [i+1 for i in k]
    df_main = pd.DataFrame(zs.transpose(), columns=new_k, index=gamma)
    #ax[0] = sns.heatmap(df_main, annot=True, alpha=1, cmap=cmap, cbar=False, ax=ax[0], vmin=zs.min(), vmax=zs.max(), fmt='.3f')
    ax[0] = sns.heatmap(df_main, annot=True, alpha=1, cmap=cmap, cbar=False, ax=ax[0], fmt='.3f')
    ax[0].invert_yaxis()
    ax[0].set( xlabel = "Fold", ylabel = "gamma")

    if True:
        figtosave=plt.figure('gamma_tosave0')
        #axtosave = sns.heatmap(df_main, annot=True, alpha=1, cmap=cmap, cbar=False, vmin=zs.min(), vmax=zs.max(), fmt='.3f')
        axtosave = sns.heatmap(df_main, annot=True, alpha=1, cmap=cmap, cbar=False, fmt='.3f')
        axtosave.invert_yaxis()
        axtosave.set_title('Dice-Sørensen score')
        axtosave.set( xlabel = "Fold", ylabel = "gamma")
        axtosave.tick_params(axis='x', rotation=0)
        figtosave.savefig('etc/out/gamma',bbox_inches='tight')
    
    ax[1].set_title('Average Dice-Sørensen')
    #cmap = sns.cubehelix_palette(start=1, light=0.8, as_cmap=True)
    cmap = sns.cubehelix_palette(start=0, dark=0.1, light=.6, reverse=False, as_cmap=True)
    avg_df = pd.DataFrame(np.mean(zs,axis=0).squeeze(), index=gamma)
    avg_df = avg_df.transpose() # transposed graph
    ax[1] = sns.heatmap(avg_df, annot=True, alpha=1, cmap=cmap, cbar=False, ax=ax[1], fmt='.3f')
    ax[1].invert_yaxis()
    ax[1].set( xlabel = None, xticklabels=[], ylabel = "gamma")
    ax[1].tick_params(axis='y', rotation=0)
    
    if True:
        #figtosave=plt.figure('gamma_tosave1',figsize=(3, 6), dpi=80, layout='tight')
        figtosave=plt.figure('gamma_tosave1',figsize=(8, 2.3), dpi=80, layout='tight') # transposed graph
        axtosave = sns.heatmap(avg_df, annot=True, alpha=1, cmap=cmap, cbar=False, fmt='.3f')
        axtosave.invert_yaxis()
        axtosave.set_title('Average Dice-Sørensen')
        #axtosave.set( xlabel = None, xticklabels=[], ylabel = "gamma")
        axtosave.set( ylabel = None, yticklabels=[], xlabel = "gamma")  # transposed graph
        axtosave.tick_params(axis='y', rotation=0)
        axtosave.tick_params(axis='x', rotation=0)
        figtosave.savefig('etc/out/avg-gamma',bbox_inches='tight')
    
    ax[2].set_title('Dice-Sørensen st.err.')
    #cmap = sns.cubehelix_palette(start=2, light=0.8, as_cmap=True)
    cmap = sns.cubehelix_palette(start=0, dark=0.15, light=.6, reverse=False, as_cmap=True)
    std_df = pd.DataFrame(np.std(zs,axis=0).squeeze()/ np.sqrt(5), index=gamma)
    std_df = std_df.transpose() # transposed graph
    ax[2] = sns.heatmap(std_df, annot=True, alpha=1, cmap=cmap, cbar=False, ax=ax[2], fmt='.3f')
    ax[2].invert_yaxis()
    ax[2].set( xlabel = None, xticklabels=[], ylabel = "gamma")
    ax[2].tick_params(axis='y', rotation=0)
    
    if True:
        #figtosave=plt.figure('gamma_tosave2',figsize=(3, 6), dpi=80, layout='tight')
        figtosave=plt.figure('gamma_tosave2',figsize=(8, 2.3), dpi=80, layout='tight') # transposed graph
        axtosave = sns.heatmap(std_df, annot=True, alpha=1, cmap=cmap, cbar=False, fmt='.3f')
        axtosave.invert_yaxis()
        axtosave.set_title('Dice-Sørensen st.err.')
        #axtosave.set( xlabel = None, xticklabels=[], ylabel = "gamma")
        axtosave.set( ylabel = None, yticklabels=[], xlabel = "gamma")  # transposed graph
        axtosave.tick_params(axis='x', rotation=0)
        axtosave.tick_params(axis='y', rotation=0)
        figtosave.savefig('etc/out/var-gamma',bbox_inches='tight')

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
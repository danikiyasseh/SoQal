#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:54:39 2020

@author: Dani Kiyasseh
"""

#%%
import os
import numpy as np
from tabulate import tabulate
import argparse

from run_experiment import train_model, make_saving_directory
from prepare_network import cnn_network_time, cnn_network_image
from prepare_miscellaneous import determine_classification_setting

#%%
""" Directory to Folder that Contains Dataset(s) """
basepath_to_data = '/mnt/SecondaryHDD'
""" Define Downstream Task """
downstream_task = 'los'
""" Original Split According to Patients """
#fraction = 0.9
""" Of the Above, Subsample Labelled Data @ Frame-Level """
labelled_fraction = 1 #0.1
""" Subsample Unlabelled Data @ Frame-Level """
unlabelled_fraction = 1 #0.05
""" Number of MC Dropouts """
dropout_samples = 5 #20
""" Initialization """
meta = False #use initialization from meta-training? #False allows you to compare to random initialization

def print_hyperparam_info(acquisition_epochs,meta,input_perturbed,downstream_dataset,classification,modalities,downstream_task,fraction,labelled_fraction,unlabelled_fraction,dropout_samples,metric,batch_size,held_out_lr,seed):
    if len(acquisition_epochs)>1:
        info = list(map(lambda x:[x],[seed,meta,input_perturbed,downstream_dataset,classification,modalities,downstream_task,fraction,labelled_fraction,unlabelled_fraction,acquisition_epochs[0],acquisition_epochs[-1],dropout_samples,metric,batch_size,held_out_lr]))
        header = ['seed','meta','consistency','downstream_dataset','classification','modalities','downstream_task','fraction','labelled_fraction','unlabelled_fraction','acquistion_start','acquistion_end','dropout_samples','metric','batchsize','lr']
    else:
        info = list(map(lambda x:[x],[seed,meta,input_perturbed,downstream_dataset,classification,modalities,downstream_task,fraction,labelled_fraction,unlabelled_fraction,dropout_samples,metric,batch_size,held_out_lr]))
        header = ['seed','meta','consistency','downstream_dataset','classification','modalities','downstream_task','fraction','labelled_fraction','unlabelled_fraction','dropout_samples','metric','batchsize','lr']
        
    info_dict = dict(zip(header,info))
    print(tabulate(info_dict,header))

#%%

def run_configurations(args):
    saved_weights_list = ['finetuned_weight']
    downstream_datasets_list = args.datasets   #list(map(str,sys.argv[1].strip('[]').split(','))) #[str(sys.argv[1]),str(sys.argv[2])]#['physionet2017','cardiology']#,'bidmc','mimic']
    print(downstream_datasets_list)
    batch_size_list = args.batchsize #list(map(int,sys.argv[2].strip('[]').split(','))) #[bs for bs in sys.argv[2]] #[256,16]#[256,256,16]
    modality_list = args.modalities #[['ecg']]#['ppg'],['ecg'],['ecg'],['ecg'],['ecg']]#,['ecg'],['ecg','ppg']]
    leads_list = args.leads #list(map(str,sys.argv[3].strip('[]').split(','))) # [str(ld) for ld in sys.argv[3]] #[None,None]#,'ii'] #None
    held_out_lr_list = args.held_out_lr #list(map(float,sys.argv[4].strip('[]').split(','))) #[lr for lr in sys.argv[4]] #[1e-4,1e-4] #5e-5 for ptb
    fraction_list = [0.5]#,0.7,0.9]
    acquisition_percent = 0.10 #0.02 #0.10 for cifar10
    meta_list = [False]#,True]
    acquisition_epochs_list = [list(np.arange(2,200,2))] #[[]] #2,200,2 for cifar 10
    test_perturbation = False #applying perturbation at test-time to evaluate robustness to perturbation
    #trial = 'abstention_penalty' #None #cold_gt' #'cold_gt'#'approach1' #None #'approach3' #None includes all first trials #Not None is extra investigation trials
    #heads = 'multi' #multi-head output of network #options: 'single' OR 'multi'
    #noise_type = None #label noise options: 'random' OR 'nearest_neighbour'
    #noise_level = 0
    #tolerance = 0.05 #how many wasted requests is an expert willing to handle
    """ ---------- Choose Network --------- """
    network_to_use = cnn_network_image #options: cnn_network_time | covid_net | cnn_network_image
    noise_level_list = [0.05,0.1,0.2,0.4,0.8] #probability of flipping ground truth labels
    #tolerance_list = [0.02,0.04,0.10,0.20,0.5] #this helps determine threshold on selection function
    tolerance_list = ['None']
    
    """ Trials and Noise for SoQal Paper """
    noise_dict = {'None':
                    {'noise_level_list':[0]},
                'random': 
                    {'noise_level_list':noise_level_list},
                'nearest_neighbour':
                    {'noise_level_list':noise_level_list}
                }
    
    trials_dict = {'abstention_penalty':
                        {'heads':'multi'},
                    'epsilon-greedy':
                        {'heads':'single'},
                    'entropy_response':
                        {'heads':'single'}}
    
    """ Trials and Noise for ALPS Paper """
#    trials_dict = {'None': #'None' | 'cold_gt'
#                    {'heads':'single'}}
#
#    noise_dict = {'None':
#                    {'noise_level_list':[0]}
#                }

    if test_perturbation == False:
        testfile_to_check = 'test_auc'
    elif test_perturbation == True:
        testfile_to_check = 'test_perturbed_auc'
    
    """ Acquisition Functions for ALPS Paper """
#    metric_list =  ['variance_ratio','variance_ratio.time','entropy','entropy.time','bald','bald.time'] #['balc_JSD.time','balc_KLD','balc_JSD.time','balc_KLD.time'] #['balc_JSD','balc_KLD','balc_JSD.time']
#    balc_metric_list = ['balc_KLD','balc_KLD.time','balc_JSD','balc_JSD.time']#[
    """ Acquisition Functions for when trial = cold_gt """
#    metric_list =  ['bald','bald.time']
#    balc_metric_list = ['balc_KLD','balc_KLD.time','balc_JSD','balc_JSD.time']
    
    """ Acquisition Functions for SoQal Paper """
    metric_list = ['bald']
    balc_metric_list = ['balc_KLD','balc_KLD.time']
    
    formulation_dict = {'mc_dropout':
                            {'acquisition': 'stochastic',
                             'input_perturbed': False,
                             'perturbation': 'deterministic',
                             'metric_list': metric_list},
                        'mc_consistency':
                            {'acquisition': 'deterministic',
                             'input_perturbed': True,
                             'perturbation': 'stochastic',
                             'metric_list': metric_list},
                        'balc':
                            {'acquisition': 'stochastic',
                             'input_perturbed': True,
                             'perturbation': 'deterministic',
                             'metric_list': balc_metric_list}
                            }
    
    max_seed = 5
    seed_list = np.arange(0,max_seed) #(2,5) #[100] for trial is something
    max_epochs = 40 #350
    """ Type of Hyperparam and Search Iterations """
    hyperparam_prefix = 'oracle_loss_lambda'
    #hyperparam_iterations = 1
    hellinger_threshold_list = [0.15]#,0.175]
    #for seed in seed_list:
        #print(seed)
    """ Iterate Over Some Iterable """
    for hellinger_threshold in hellinger_threshold_list:
        """ Start - Activate for SoQal - Deactive for ALPS """
        lambda1 = 1
        #hyperparam = '_'.join((hyperparam_prefix,str(lambda1)))
        """ End """
        hyperparam = ''

        for trial,trial_info in trials_dict.items():
            heads = trial_info['heads']
            for tolerance in tolerance_list:
                for noise_type,noise_info in noise_dict.items():
                    noise_level_list = noise_info['noise_level_list']
                    for noise_level in noise_level_list:
                        for downstream_dataset,batch_size,modalities,leads,held_out_lr in zip(downstream_datasets_list,batch_size_list,modality_list,leads_list,held_out_lr_list):
                            classification = determine_classification_setting(downstream_dataset,trial)
                            for fraction in fraction_list:
                                #for modalities in modality_list:
                                for meta in meta_list:
                                    for acquisition_epochs in acquisition_epochs_list:
                                        #print(acquisition_epochs)
                                        if len(acquisition_epochs) == 0: #no active learning - baseline
                                            metric,input_perturbed = 'filler', False
                                            if 'train' in phases:
                                                for seed in seed_list:
                                                    save_path_dir, seed = make_saving_directory(phases,downstream_dataset,fraction,modalities,meta,acquisition_epochs,metric,seed,max_seed,leads=leads,trial=trial,hyperparam=hyperparam,tolerance=tolerance,noise_type=noise_type,noise_level=noise_level)
                                                    print(save_path_dir)
                                                    if save_path_dir == 'do not train' and 'train' in phases:
                                                        print(save_path_dir)
                                                        continue
                                                    print_hyperparam_info(acquisition_epochs,meta,input_perturbed,downstream_dataset,classification,modalities,downstream_task,fraction,labelled_fraction,unlabelled_fraction,dropout_samples,metric,batch_size,held_out_lr,seed)
                                                    finetuned_model, report, confusion, _, _ = train_model(basepath_to_data,dropout_samples,network_to_use,save_path_dir,seed,meta,metric,acquisition_epochs,classification,batch_size,held_out_lr,fraction,unlabelled_fraction,labelled_fraction,modalities,saved_weights_list,phases,downstream_task,downstream_dataset,acquisition_percent=acquisition_percent,lambda1=lambda1,input_perturbed=input_perturbed,leads=leads,trial=trial,mixture=False,weighted_sampling=False,heads=heads,noise_type=noise_type,noise_level=noise_level,num_epochs=max_epochs)
                                            elif 'test' in phases: #only go through available seeds
                                                for s in range(1):
                                                    save_path_dir, _ = make_saving_directory(phases,downstream_dataset,fraction,modalities,meta,acquisition_epochs,metric,s,max_seed,leads=leads,trial=trial,hyperparam=hyperparam,tolerance=tolerance,noise_type=noise_type,noise_level=noise_level)
                                                    if save_path_dir == 'do not test':
                                                        continue
                                                    path_above = '/'.join(save_path_dir.split('/')[:-1])
                                                    seed_list = os.listdir(path_above) #actual seed_list that exists
                                                    print(path_above,seed_list)
                                                    for seed in seed_list:
                                                        save_path_dir = os.path.join(path_above,seed)
                                                        """ Do Not Overwrite Previous Test Result """
                                                        if testfile_to_check in os.listdir(save_path_dir):
                                                            continue
                                                        print(save_path_dir)
                                                        print_hyperparam_info(acquisition_epochs,meta,input_perturbed,downstream_dataset,classification,modalities,downstream_task,fraction,labelled_fraction,unlabelled_fraction,dropout_samples,metric,batch_size,held_out_lr,seed)
                                                        finetuned_model, report, confusion, _, _ = train_model(basepath_to_data,dropout_samples,network_to_use,save_path_dir,seed,meta,metric,acquisition_epochs,classification,batch_size,held_out_lr,fraction,unlabelled_fraction,labelled_fraction,modalities,saved_weights_list,phases,downstream_task,downstream_dataset,acquisition_percent=acquisition_percent,input_perturbed=test_perturbation,leads=leads,trial=trial,mixture=False,weighted_sampling=False,heads=heads,noise_type=noise_type,noise_level=noise_level,num_epochs=max_epochs)
                                        else: #active learning path
                                            for formulation,training_info in formulation_dict.items(): #MC Dropout, MC Consistency, and BALC Path
                                                acquisition = training_info['acquisition']
                                                input_perturbed = training_info['input_perturbed']
                                                perturbation = training_info['perturbation']
                                                metric_list = training_info['metric_list']
                                                
                                                for m,metric in enumerate(metric_list):
                                                    if 'train' in phases:
                                                        for seed in seed_list:
                                                            save_path_dir, seed = make_saving_directory(phases,downstream_dataset,fraction,modalities,meta,acquisition_epochs,metric,seed,max_seed,acquisition,input_perturbed,perturbation,leads=leads,trial=trial,hyperparam=hyperparam,tolerance=tolerance,noise_type=noise_type,noise_level=noise_level,hellinger_threshold=hellinger_threshold)
                                                            #print(formulation,metric,seed)
                                                            print(save_path_dir)
                                                            if save_path_dir == 'do not train' and 'train' in phases:
                                                                print(save_path_dir)
                                                                continue
                                                            print_hyperparam_info(acquisition_epochs,meta,input_perturbed,downstream_dataset,classification,modalities,downstream_task,fraction,labelled_fraction,unlabelled_fraction,dropout_samples,metric,batch_size,held_out_lr,seed)
                                                            finetuned_model, report, confusion, _, _ = train_model(basepath_to_data,dropout_samples,network_to_use,save_path_dir,seed,meta,metric,acquisition_epochs,classification,batch_size,held_out_lr,fraction,unlabelled_fraction,labelled_fraction,modalities,saved_weights_list,phases,downstream_task,downstream_dataset,acquisition_percent=acquisition_percent,hellinger_threshold=hellinger_threshold,lambda1=lambda1,acquisition=acquisition,input_perturbed=input_perturbed,perturbation=perturbation,leads=leads,trial=trial,mixture=False,weighted_sampling=False,heads=heads,noise_type=noise_type,noise_level=noise_level,num_epochs=max_epochs)
                                                    elif 'test' in phases: #only go through available seeds
                                                        for s in range(1):
                                                            save_path_dir, seed = make_saving_directory(phases,downstream_dataset,fraction,modalities,meta,acquisition_epochs,metric,s,max_seed,acquisition,input_perturbed,perturbation,leads=leads,trial=trial,hyperparam=hyperparam,tolerance=tolerance,noise_type=noise_type,noise_level=noise_level,hellinger_threshold=hellinger_threshold)
                                                            path_above = '/'.join(save_path_dir.split('/')[:-1])
                                                            seed_list = os.listdir(path_above) #actual seed_list that exists
                                                            #print(seed_list)
                                                            for seed in seed_list:
                                                                if 'seed' in seed:
                                                                    save_path_dir = os.path.join(path_above,seed)
                                                                    """ Do Not Overwrite Previous Test Result """
                                                                    if testfile_to_check in os.listdir(save_path_dir):
                                                                        continue
                                                                    print_hyperparam_info(acquisition_epochs,meta,input_perturbed,downstream_dataset,classification,modalities,downstream_task,fraction,labelled_fraction,unlabelled_fraction,dropout_samples,metric,batch_size,held_out_lr,seed)
                                                                    #print(save_path_dir)
                                                                    finetuned_model, report, confusion, _, _ = train_model(basepath_to_data,dropout_samples,network_to_use,save_path_dir,seed,meta,metric,acquisition_epochs,classification,batch_size,held_out_lr,fraction,unlabelled_fraction,labelled_fraction,modalities,saved_weights_list,phases,downstream_task,downstream_dataset,acquisition_percent=acquisition_percent,acquisition=acquisition,input_perturbed=test_perturbation,leads=leads,trial=trial,mixture=False,weighted_sampling=False,heads=heads,noise_type=noise_type,noise_level=noise_level,num_epochs=max_epochs)

#%%
if __name__ == '__main__':
    #print('Skipping Training')
    #phases = ['train','val']#['train','val'] 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--phases',nargs='+',default=['train','val'],required=True,help='phases')
    parser.add_argument("--datasets",nargs='+',default=['physionet'],required=True,help="dataset list")
    parser.add_argument("--batchsize",nargs='+',default=[256],required=True,type=int,help="batchsize list")
    parser.add_argument("--modalities",nargs='+',default=['ecg'],required=True,help="modalities list")
    parser.add_argument("--leads",nargs='+',default=['i'],required=True,help="leads list")
    parser.add_argument("--held_out_lr",nargs='+',default=[1e-4],required=True,type=float,help="held out lr list")
    args = parser.parse_args()
    
    phases = args.phases #['test']#['train','val'] #['test']
    run_configurations(args)


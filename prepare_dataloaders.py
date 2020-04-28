#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:22:35 2020

@author: Dani Kiyasseh
"""
#%%

""" Functions in this Script 
    1) load_inputs_and_outputs
    2) load_initial_data
    3) load_dataloaders_list_active
    4) check_dataset_allignment 
"""

#%%
import pickle
import os
from prepare_dataset import my_dataset_direct
from torch.utils.data import DataLoader
#%%
def load_inputs_and_outputs(basepath,dataset_name,leads='i',cl_scenario=None):
    
    if dataset_name == 'bidmc':
        path = os.path.join(basepath,'BIDMC v1')
        extension = 'heartpy_'
    elif dataset_name == 'physionet':
        path = os.path.join(basepath,'PhysioNet v2')
        extension = 'heartpy_'
    elif dataset_name == 'mimic':
        shrink_factor = str(0.1)
        path = os.path.join(basepath,'MIMIC3_WFDB','frame-level',shrink_factor)
        extension = 'heartpy_'
    elif dataset_name == 'cipa':
        lead = ['II','aVR']
        path = os.path.join(basepath,'cipa-ecg-validation-study-1.0.0','leads_%s' % lead)
        extension = ''
    elif dataset_name == 'cardiology':
        classes = 'all'
        path = os.path.join(basepath,'CARDIOL_MAY_2017','patient_data','%s_classes' % classes)
        extension = ''
    elif dataset_name == 'physionet2017':
        path = os.path.join(basepath,'PhysioNet 2017','patient_data')
        extension = ''
    elif dataset_name == 'tetanus':
        path = '/media/scro3517/TertiaryHDD/new_tetanus_data/patient_data'
        extension = ''
    elif dataset_name == 'ptb':
        leads = [leads]
        path = os.path.join(basepath,'ptb-diagnostic-ecg-database-1.0.0','patient_data','leads_%s' % leads)
        extension = ''  
    elif dataset_name == 'fetal':
        abdomen = leads #'Abdomen_1'
        path = os.path.join(basepath,'non-invasive-fetal-ecg-arrhythmia-database-1.0.0','patient_data',abdomen)
        extension = ''
    elif dataset_name == 'physionet2016':
        path = os.path.join(basepath,'classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0')
        extension = ''
    elif dataset_name == 'physionet2020':
        basepath = '/mnt/SecondaryHDD'
        leads = [leads]
        path = os.path.join(basepath,'PhysioNetChallenge2020_Training_CPSC','Training_WFDB','patient_data','leads_%s' % leads)
        extension = ''
    elif dataset_name == 'chapman':
        basepath = '/mnt/SecondaryHDD'
        leads = leads
        path = os.path.join(basepath,'chapman_ecg','leads_%s' % leads)
        extension = ''
    elif dataset_name == 'cifar10':
        basepath = '/mnt/SecondaryHDD'
        leads = ''
        path = os.path.join(basepath,'cifar-10-python/cifar-10-batches-py')
        extension = '' 

    if cl_scenario == 'Class-IL':
        dataset_name = dataset_name + '_' + 'mutually_exclusive_classes'

    """ Dict Containing Actual Frames """
    with open(os.path.join(path,'frames_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as f:
        input_array = pickle.load(f)
    """ Dict Containing Actual Labels """
    with open(os.path.join(path,'labels_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as g:
        output_array = pickle.load(g)
    
    return input_array,output_array,path

def load_initial_data(basepath_to_data,phases,classification,fraction,inferences,unlabelled_fraction,labelled_fraction,batch_size,modality,acquired_indices,acquired_labels,downstream_task,modalities,dataset_name,leads='ii',mixture='independent'):    
    """ Control augmentation at beginning of training here """ 
    resize = False
    affine = False
    rotation = False
    color = False    
    perform_cutout = False
    operations = {'resize': resize, 'affine': affine, 'rotation': rotation, 'color': color, 'perform_cutout': perform_cutout}    
    shuffles = {'train1':True,
                'train2':False,
                'val': False,
                'test': False}
    
    fractions = {'fraction': fraction,
                 'labelled_fraction': labelled_fraction,
                 'unlabelled_fraction': unlabelled_fraction}
    
    acquired_items = {'acquired_indices': acquired_indices,
                      'acquired_labels': acquired_labels}
    
    dataset_list = [{phase:my_dataset_direct(basepath_to_data,dataset_name,phase,inference,fractions,acquired_items,modalities=modalities,task=downstream_task,leads=leads) for phase,inference in zip(phases,inferences)}]                                        
    
    if 'train' in phases:
        check_dataset_allignment(mixture,dataset_list)
        
    dataloaders_list = [{phase:DataLoader(dataset[phase],batch_size=batch_size,shuffle=shuffles[phase],drop_last=False) for phase in phases} for dataset in dataset_list]
    print(len(dataloaders_list))
    
    return dataloaders_list,operations

def load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,dataset_name,leads='ii',input_perturbed=False):   
    shuffles = {'train1':True,
                'train2':False,
                'val': False,
                'test': False}
        
    fractions = {'fraction': fraction,
                 'labelled_fraction': labelled_fraction,
                 'unlabelled_fraction': unlabelled_fraction}
    
    acquired_items = {'acquired_indices': acquired_indices,
                      'acquired_labels': acquired_labels}
    
    dataset_list = [{phase:my_dataset_direct(dataset_name,phase,inference,fractions,acquired_items,modalities=modalities,task=downstream_task,input_perturbed=input_perturbed,leads=leads) for phase,inference in zip(phases,inferences)}]                                        
    
    check_dataset_allignment(mixture,dataset_list)
    
    #print('Batchsize: %i' % batch_size)
    if input_perturbed == False:
        print('Active Dataloaders!')
    elif input_perturbed == True:
        print('Active Perturbed Dataloaders!')
        
    dataloaders_list = [{phase:DataLoader(dataset[phase],batch_size=batch_size,shuffle=shuffles[phase],drop_last=False) for phase in phases} for dataset in dataset_list]

    return dataloaders_list

def check_dataset_allignment(mixture,dataset_list):
    if mixture:
        length_prev = 0 #starter
        for i in range(len(dataset_list)):
            length_curr = len(dataset_list[i]['train'])
            if i != 0:
                if length_curr != length_prev:
                    print('Caution! Datasets are not alligned')
                    exit()
            length_prev = length_curr
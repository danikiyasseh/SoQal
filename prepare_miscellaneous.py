#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:46:09 2020

@author: Dani Kiyasseh
"""
#%%

""" Functions in this Script 
    1) change_lr
    2) change_weight_decay
    3) obtain_loss_function
    4) obtain_predictions
"""

#%%
import torch
import numpy as np
import torch.nn as nn
from operator import itemgetter

#%%

def change_lr(epoch_count,optimizer):
    """ Manually change (multiplicative) learning rate at pre-defined epochs """
    transition_epochs = None
    scale = 0.5
    if transition_epochs is not None:
        if epoch_count == transition_epochs[0]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*scale
                print('LR: %.5f' % param_group['lr'])

def change_weight_decay(epoch_count,optimizer):
    """ Manually change (additive) weight decay at pre-defined epochs """
    transition_epochs = None #[8]
    scale = 1e-1
    if transition_epochs is not None:
        if epoch_count == transition_epochs[0]:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = param_group['weight_decay'] + scale
                print('Weight Decay: %.5f' % param_group['weight_decay'])

def obtain_loss_function(phase,classification,dataloaders_list,pos_weight=1,imbalance_penalty=None):
    if classification is not None:
        nclasses = classification.split('-')[0]
    
    if 'train' in phase:
        """ Dataloader - Image-Based """ 
        #train_indices = dataloaders_list[0]['train'].batch_sampler.sampler.data_source.indices
        #all_outputs = dataloaders_list[0]['train'].batch_sampler.sampler.data_source.outputs

        all_outputs = dataloaders_list[0]['train1'].batch_sampler.sampler.data_source.label_array

        if imbalance_penalty == True:
            """ Obtain Weights for Optimizer (Class Imbalance) """            

            train_outputs = list(itemgetter(*train_indices)(all_outputs))
            val,bins = np.histogram(train_outputs,nclasses)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
            loss_weight = torch.tensor(max(val)/val,dtype=torch.float,device=device)
            """ Define Optimizer """
            if classification is not None and classification != '2-way':
                criterion = nn.CrossEntropyLoss(pos_weight=loss_weight)
                criterion_single = nn.CrossEntropyLoss(pos_weight=loss_weight,reduction='none')
            elif classification == '2-way':
                criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
                criterion_single = nn.BCEWithLogitsLoss(pos_weight=loss_weight,reduction='none')                
        else:
            if classification is not None and classification != '2-way':
                criterion = nn.CrossEntropyLoss()
                criterion_single = nn.CrossEntropyLoss(reduction='none')          
            elif classification == '2-way':
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
                criterion_single = nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(pos_weight)) 
            elif classification is None:
                criterion = nn.MSELoss()
                criterion_single = nn.MSELoss(reduction='none')
                
        """ Running Loss per Sample """
        keys = np.arange(len(all_outputs))
        values = [[] for _ in range(len(keys))]
        per_sample_loss_dict = dict(zip(keys,values))
        
        return per_sample_loss_dict, criterion, criterion_single
    else:
        if classification is not None and classification != '2-way':
            criterion = nn.CrossEntropyLoss()
            criterion_single = nn.CrossEntropyLoss(reduction='none')          
        elif classification == '2-way':
            criterion = nn.BCEWithLogitsLoss()
            criterion_single = nn.BCEWithLogitsLoss(reduction='none') 
        elif classification is None:
            criterion = nn.MSELoss()
            criterion_single = nn.MSELoss(reduction='none')
            
        return criterion, criterion_single

def obtain_predictions(output_probs,device,classification):
    if classification is not None and classification != '2-way':
        _,preds = torch.max(output_probs,1)
    elif classification == '2-way':
        """ May have to Subtract Mean from Outputs Before Taking Sigmoid """
        #preds = torch.where(torch.sigmoid(outputs)>0.5,torch.tensor(1,device=device),torch.tensor(0,device=device))
        preds = torch.where(output_probs>0.5,torch.tensor(1,device=device),torch.tensor(0,device=device))
    return preds

def determine_classification_setting(dataset_name,trial):
    """ This is used to determine loss function i.e. C.E.L. or BCE """
    if dataset_name == 'physionet':
        classification = '5-way'
    elif dataset_name == 'bidmc':
        classification = '2-way'
    elif dataset_name == 'mimic': #change this accordingly
        classification = '2-way'
    elif dataset_name == 'cipa':
        classification = '7-way'
    elif dataset_name == 'cardiology':
        classification = '12-way'
    elif dataset_name == 'physionet2017':
        classification = '4-way'
    elif dataset_name == 'tetanus':
        classification = '2-way'
    elif dataset_name == 'ptb':
        classification = '2-way'
    elif dataset_name == 'fetal':
        classification = '2-way'
    elif dataset_name == 'physionet2020':
        classification = '2-way' #binary classification in multilabel scenario
    elif dataset_name == 'uci_emg':
        classification = '6-way'
    elif dataset_name == 'covid19':
        classification = '2-way'
    elif dataset_name == 'cifar10':
        classification = '10-way'
    
    print(dataset_name)
    
    return classification
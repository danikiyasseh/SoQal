#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:42:15 2020

@author: Dani Kiyasseh
"""
#%%

""" Functions in this Script 
    1) load_initial_model
    2) load_models_list
    3) check_mismatch_and_load_weights
    4) perturb_weights
"""

#%%
import os
import torch
import torch.optim as optim
import copy
import numpy as np

from prepare_dataloaders import load_inputs_and_outputs

#%%

def load_initial_model(meta,classification,cnn_network,cl_strategy,phases,save_path_dir,saved_weights_list,held_out_lr,mixture='independent',colors_dirs=None,continual_setting=False,dataset_name=None,bptt_steps=None,heads='multi',setting='Domain-IL',new_task_info=None,task_instance_importance=False,cl_scenario=None,trial=None):
    """ Load models with maml weights """
    dropout_type = 'drop1d' #options: | 'drop1d' | 'drop2d'
    p1,p2,p3 = 0.1,0.1,0.1 #initial dropout probabilities #0.1, 0.1, 0.1
    """ directory of meta-learned weights """
    #torch.manual_seed(0) #must be done before each instantiation 
    nmix = len(saved_weights_list)
    if continual_setting == True:
        models_list = [cnn_network(dropout_type,p1,p2,p3,dataset_name,hyperattention_type=cl_strategy,bptt_steps=bptt_steps,heads=heads,setting=setting,trial=trial) for _ in range(nmix)]
        print('Continual Setting!')
    else:
        models_list = [cnn_network(dropout_type,p1,p2,p3,classification,heads=heads) for _ in range(nmix)]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    print(device)

    if mixture == 'dependent':
        """ Individual Models get their own regularization etc. """
        dropout_list = ['drop2d','drop2d','drop2d','drop2d'] #['drop2d','drop2d','drop1d','drop2d'] #based on which and order of individual models used  
        p1_list = [0.4,0.4,0.4,0.4]#[0.3,0.2,0.2,0.3]
        p2_list = [0.4,0.4,0.4,0.4]#[0.3,0.2,0.2,0.2]
        p3_list = [0.4,0.4,0.4,0.4]#[0.0,0.0,0.0,0.2]
        models_list = [cnn_network(dropout_type,p1,p2,p3,classification) for dropout_type,p1,p2,p3 in zip(dropout_list,p1_list,p2_list,p3_list)]

    """ Inference Without Meta Learning """
    if 'test' in phases and len(phases) == 1 or 'val' in phases and len(phases) == 1:
        parameters_list = [torch.load(os.path.join(save_path_dir,saved_weights))[0] for saved_weights in saved_weights_list]
        [model.load_state_dict(parameters) for model,parameters in zip(models_list,parameters_list)]
        print('Finetuned Weights Loaded!')
    
    [model.to(device) for model in models_list]
    
    if task_instance_importance == True:
        """  Task-Instance Parameters """
        new_task_datasets = new_task_info['new_task_datasets']
        new_task_modalities = new_task_info['new_task_modalities']
        new_task_leads = new_task_info['new_task_leads']
        new_task_fractions = new_task_info['new_task_fractions']
        new_task_class_pairs = new_task_info['new_task_class_pairs']
        task_instance_params_dict = dict() #ParameterDict?
        print('Preparing Task-Instance Params')
        for task,modality,leads,fraction,class_pair in zip(new_task_datasets,new_task_modalities,new_task_leads,new_task_fractions,new_task_class_pairs):
            inputs,outputs,path = load_inputs_and_outputs(task,leads,cl_scenario)
            modality = modality[0]
            
            if cl_scenario == 'Class-IL' or cl_scenario == 'Time-IL' or (cl_scenario == 'Task-IL' and 'chapman' in task):
                header = class_pair
            else:
                header = 'labelled'
            
            nsamples = outputs[modality][fraction]['train'][header].shape[0]
            print(nsamples)
            #task_parameters = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(nsamples)]) #check this out
            name = '-'.join((task,modality,str(fraction),leads,class_pair))
            #task_instance_params_dict[name] = task_parameters.to(device) #.to(device)
            
            """ Added New """
            task_parameters = torch.ones(nsamples,requires_grad=True,device=device) #zeros if using exp, ones otherwise
            task_instance_params_dict[name] = task_parameters
            """ Ended New """
            
        #task_instance_params = [param for task,list_of_params in task_instance_params_dict.items() for param in list_of_params.parameters()]
        print('Finished!')
    else:
        task_instance_params_dict = None
        #task_instance_params = []
    #num_classes = 3
    model_params = [list(model.parameters()) for model in models_list] #list of params from each model
    model_params = [param for net in model_params for param in net] #+ task_instance_params #flattening out list
    #print(model_params)
    
    if mixture == 'dependent' and 'train' in phases:
        #creating mixture coefficients as 'leaf variables' to be accepted by optimizer 
        #mix_coefs = torch.tensor(np.repeat(1/(nmix),nmix),requires_grad=True,dtype=torch.float,device="cuda") #initializing mix coefs naively
        model_params.append(mix_coefs)
    elif mixture == 'dependent' and 'test' in phases:
        model_params.append(mix_coefs)        
    else:
        mix_coefs = None
    
    optimizer = optim.Adam(model_params,lr=held_out_lr,weight_decay=0) #shouldn't load this again - will lose running average of gradients and so forth
    if task_instance_importance == True:
        """ Added New """
        param_optimizer = optim.Adam(list(task_instance_params_dict.values()),lr=0.0001,weight_decay=0)
        optimizer = (optimizer,param_optimizer)
    """ Ended New """
    
    return models_list,mix_coefs,optimizer,device,task_instance_params_dict

def load_models_list(epoch_count,classification,cnn_network,device,models_list,test_colourmap=None): #weights are either maml weights OR temp weights before changing dropout | models_list and mix coefs are fillers initially
    """ Load Models Mid-Training for Augmentation Purposes """
    transition_epochs = None #epochs at which changes occur | None | [17]
    if transition_epochs is not None:
        print('HELLO')
        if epoch_count == transition_epochs[0]:
            dropout_type = 'drop1d'
            p1,p2,p3 = 0,0,0 #more aggressive dropout probabilities 
            print('Dropout: %s' % str([p1,p2,p3]))
    
        if epoch_count in transition_epochs: #helps when there exists more than one transition 
            temp_weights_list = [copy.deepcopy(model.state_dict()) for model in models_list] #hold onto weights for reloading in two lines 
            #models_list = [cnn_network(dropout_type,p1,p2,p3) for _ in range(len(color_dirs))] #temp weights should be in meta directory
            models_list = [cnn_network(dropout_type,p1,p2,p3,classification) for _ in range(len(test_colourmap))]
            [model.load_state_dict(temp_weights) for model,temp_weights in zip(models_list,temp_weights_list)]
            #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
            [model.to(device) for model in models_list]
        else: #prevents reloading the model unnecessarily (think of this as identity function)
            models_list = models_list
                
    return models_list

def check_mismatch_and_load_weights(models_list,parameters_list):
    model_list_state_dict = [model.state_dict() for model in models_list]
    parameters_list_state_dict = parameters_list #[parameters.state_dict() for parameters in parameters_list]
    for model_dict,parameters_dict in zip(model_list_state_dict,parameters_list_state_dict):
        for (orig_name,orig_param),(new_name,new_param) in zip(model_dict.items(),parameters_dict.items()):
            if orig_name == new_name and orig_param.shape == new_param.shape:
                orig_param.data.copy_(new_param) # underscore is vital for permanent copy

def perturb_weights(parameters_list,perturbation_dict,device):
    alpha = perturbation_dict['alpha']
    beta = perturbation_dict['beta']
    dim = perturbation_dict['dim']
    for parameters in parameters_list:
        for param_name,param in parameters.items():
            if 'batchnorm' not in param_name:
                normal_pdf = torch.distributions.normal.Normal(0,1)
                param_magnitude = torch.norm(param,p='fro')
                random_matrix1 = normal_pdf.sample(param.shape)
                random_matrix1_magnitude = torch.norm(random_matrix1,p='fro')
                perturbation1 = random_matrix1 * (param_magnitude / random_matrix1_magnitude)
                perturbation1 = perturbation1.to(device)
                if dim == '2d':
                    random_matrix2 = normal_pdf.sample(param.shape)
                    random_matrix2_magnitude = torch.norm(random_matrix2,p='fro')
                    perturbation2 = random_matrix2 * (param_magnitude / random_matrix2_magnitude)
                    perturbation2 = perturbation2.to(device)
                    new_param_values = param + (alpha * perturbation1) + (beta * perturbation2)
                elif dim == '1d':
                    new_param_values = param + (alpha * perturbation1)
                    
                param.data.copy_(new_param_values) 
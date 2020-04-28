#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:18:38 2020

@author: Dani Kiyasseh
"""

#%%
import pickle
from sklearn.mixture import GaussianMixture
import os
import numpy as np
from scipy.special import expit, softmax
import torch
import copy
from tensorboardX import SummaryWriter
writer = SummaryWriter(logdir='finetuning_runs')

from prepare_dataloaders import load_initial_data, load_dataloaders_list_active, load_inputs_and_outputs
from prepare_models import load_initial_model, load_models_list
from prepare_miscellaneous import obtain_loss_function, change_lr, change_weight_decay
from perform_training import one_epoch

from prepare_acquisition_functions import update_acquisition_dict, acquisition_function, perform_MC_sampling

#%%
def train_model(basepath_to_data,dropout_samples,network_to_use,save_path_dir,seed,meta,metric,acquisition_epochs,classification,batch_size,held_out_lr,fraction,unlabelled_fraction,labelled_fraction,modalities,saved_weights_list,phases,downstream_task,downstream_dataset,acquisition_percent=0.02,hellinger_threshold=0.15,lambda1=1,acquisition=None,input_perturbed=False,perturbation=None,leads='ii',trial=None,mixture=False,weighted_sampling=False,heads='single',noise_type=None,noise_level=0,tolerance=None,num_epochs=150):
    """ Training and Validation For All Epochs """
    best_loss = float('inf')
    modalities = [modalities]
    auc_dict = dict()
    acc_dict = dict()
    loss_dict = dict()
    if 'test' not in phases:
        phases = ['train1','val']
        inferences = [False,False]
    else:
        inferences = [False]
    
    for phase in phases:
        acc_dict[phase] = []
        loss_dict[phase] = []
        auc_dict[phase] = []
    stop_counter = 0
    patience = 2 #for early stopping criterion #15 for time-series, #2 for images
    epoch_count = 0
    cl_strategy = None
    models_list,mix_coefs,optimizer,device,_ = load_initial_model(meta,classification,network_to_use,cl_strategy,phases,save_path_dir,saved_weights_list,held_out_lr,continual_setting=False,heads=heads)
    
    if noise_type == 'None':
        nn_labels = None
    else:
        _,_,path = load_inputs_and_outputs(basepath_to_data,downstream_dataset,leads)
        with open(os.path.join(path,'NN_labels.pkl'),'rb') as f:
            nn_labels = pickle.load(f)
    
    """ Running List of Indices to Acquire During Training """
    oracle_asks = []
    proportion_wasted = []
    acquired_indices = [] #indices of the unlabelled data
    acquired_labels = dict() #network labels of the unlabelled data
    acquired_modalities = dict() #modalities of unlabelled data
    acquired_gt_labels = dict() #ground truth labels of the unlabelled for analysis later
    if 'time' in metric:
        acquisition_metric_dict = dict()
        full_dict_for_saving = dict() #dict that keeps acquired unlabelled scores for saving purposes
    
    dataloaders_list,operations = load_initial_data(basepath_to_data,phases,classification,fraction,inferences,unlabelled_fraction,labelled_fraction,batch_size,modalities,acquired_indices,acquired_labels,downstream_task,modalities,downstream_dataset,leads=leads)

    """ Obtain Number of Labelled Samples """
    #total_labelled_samples = len(dataloaders_list[0]['train1'].batch_sampler.sampler.data_source.label_array)
    """ Instantiate Hyperparam Dict Based on Running Average of Zero One Losses """
    hyperparam_dict = None #dict(zip(np.arange(total_labelled_samples),[0 for _ in range(total_labelled_samples)]))
    
    #train_scoring_function = 0 #needed for WeightedSampler 
    while stop_counter <= patience and epoch_count < num_epochs:
        if 'train' in phases or 'val' in phases:
            print('Epoch %i/%i' % (epoch_count,num_epochs-1))
            print('-' * 10)

            """ Load Model with Potential Network Changes Mid-Training """
            models_list = load_models_list(epoch_count,classification,network_to_use,device,models_list)
            """ Load DataLoader with Potential Augmentation Mid-Training """
            #dataloaders_list, operations = load_dataloaders_list(epoch_count,test_dim,dataloaders_list,test_colors_list,batch_size,weighted_sampling,train_scoring_function,operations)  
            
#            """ Dataloader - Image-Based """
#            dataloaders_list, operations = load_dataloaders_list(epoch_count,classification,mixture,test_representation,test_order,test_colourmap,test_dim,test_task,dataloaders_list,batch_size,modality,weighted_sampling,train_scoring_function,operations)
#            """ Dataloader - Image-Based """
            
            if len(acquisition_epochs) == 0 and epoch_count == 0: #for normal training path - nothing funky 
                    phases = ['train1','val']
                    inferences = [False,False]
                    dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,leads=leads)
            elif len(acquisition_epochs) > 0:
                """ Epochs to Perform Acquisition At """
                if 'time' not in metric:
                    if epoch_count in acquisition_epochs:
                        phases = ['train1','val','train2']
                        inferences = [False,False,True]
                        dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,leads=leads)
                        if input_perturbed == True:
                            #""" This Seed Ensures Perturbation is Same Across MC Passes But Different For Different Epochs - CONFIRMED """
                            #np.random.seed(epoch_count)
                            """ For now - this is just a filler - less flexibility """
                            perturbed_dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,leads,input_perturbed)
                        #dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_subsample_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,downstream_task,dataset_name='mimic')
                    else:
                        """ Ensure No Inference is Performed for Other Epochs """
                        #print(acquired_indices)
                        if 'train2' in phases: #if last epoch was an acquisition one, then change for the rest until next acquisition is seen
                            phases = ['train1','val']
                            inferences = [False,False]
                            dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,leads=leads)
                            """ I commented out the 2 lines December 20 2019 """
                            #if input_perturbed == True:
                            #    perturbed_dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,input_perturbed)
                            perturbed_dataloaders_list = None
                        #else:
                        #    perturbed_dataloaders_list = None
                else:
                    """ Time in Metric ==> MC on Every Epoch """
                    phases = ['train1','val','train2']
                    inferences = [False,False,True]
                    #print('Acquired Indices')
                    #print(acquired_indices)
                    dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,leads=leads)
                    if input_perturbed == True:
                        #np.random.seed(epoch_count)
                        perturbed_dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,leads,input_perturbed)
            
            """ Obtain Original Number of Unlabelled Samples to Determine When to Stop Acquisition """
            if len(acquisition_epochs) > 0:
                if 'train2' in phases and True in inferences:
                    if 'time' in metric:
                        if epoch_count == 0: #only initial unlabelled data size is needed
                            total_unlabelled_samples = len(dataloaders_list[0]['train2'].batch_sampler.sampler.data_source.label_array)
                    elif 'time' not in metric:
                        if epoch_count == acquisition_epochs[0]:
                            total_unlabelled_samples = len(dataloaders_list[0]['train2'].batch_sampler.sampler.data_source.label_array)
                    print(total_unlabelled_samples)
            
            """ Expand HyperParam Dict to Account for Unlabelled Data To be Added """
            #if 'train2' in phases and epoch_count == acquisition_epochs[0]:
            #    total_unlabelled_samples = len(dataloaders_list[0]['train2'].batch_sampler.sampler.data_source.label_array)
            #    extra_hyperparam_dict = dict(zip(np.arange(total_labelled_samples,total_labelled_samples + total_unlabelled_samples),[0 for _ in range(total_unlabelled_samples)]))
            #    hyperparam_dict = {**hyperparam_dict,**extra_hyperparam_dict}
            
            """ Change LR mid-training """
            change_lr(epoch_count,optimizer)
            """ Change Weight Decay mid-training """
            change_weight_decay(epoch_count,optimizer)
        elif 'test' in phases:
            print('Test Set')
            if input_perturbed == True:
                dataloaders_list = load_dataloaders_list_active(classification,fraction,inferences,unlabelled_fraction,labelled_fraction,acquired_indices,acquired_labels,mixture,dataloaders_list,batch_size,phases,modalities,downstream_task,downstream_dataset,leads=leads,input_perturbed=input_perturbed)
                print('Perturbing Inputs!')
        
        """ ACTUAL TRAINING AND EVALUATION """
        print(inferences)
        for phase,inference in zip(phases,inferences):
            if 'train1' in phase:
                [model.train() for model in models_list]
                per_sample_loss_dict, criterion, criterion_single = obtain_loss_function(phase,classification,dataloaders_list)
            elif 'train2' in phase:
                if acquisition == 'deterministic':
                    """ Ensures no Dropout Mask is Applied """
                    [model.eval() for model in models_list]
                elif acquisition == 'stochastic':
                    [model.train() for model in models_list]                    
                per_sample_loss_dict, criterion, criterion_single = obtain_loss_function(phase,classification,dataloaders_list)                
            elif phase == 'val' or phase == 'test':
                [model.eval() for model in models_list]
                criterion, criterion_single = obtain_loss_function(phase,classification,dataloaders_list)
                #print('lr: %.6f' % optimizer.param_groups[-1]['lr'])
            
            if 'train' in phase and inference == True:
                """ Perform Inference T Times i.e. MC Dropout Implementation """
                """ ADDED NEW - Jan 14th """
                if len(acquired_indices) != total_unlabelled_samples:
                    print(len(acquired_indices),total_unlabelled_samples)
                    if acquisition == 'stochastic':
                        print('Clean Input MC')
                        posterior_dict_new, modality_dict_new, gt_labels_dict_new,_ = perform_MC_sampling(dropout_samples,save_path_dir,seed,epoch_count,batch_size,fraction,modalities,downstream_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device,trial=trial,leads=leads,lambda1=lambda1)
    
                        if input_perturbed == True:
                            """ Perturbed Input Path """
                            print('Perturbed Input MC')
                            perturbed_posterior_dict_new, perturbed_modality_dict_new, _, _ = perform_MC_sampling(dropout_samples,save_path_dir,seed,epoch_count,batch_size,fraction,modalities,downstream_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,perturbed_dataloaders_list,models_list,mix_coefs,optimizer,device,inferences,acquired_indices,acquired_labels,input_perturbed,trial=trial,leads=leads,lambda1=lambda1)
                        else:
                            perturbed_posterior_dict_new = None 
                            
                    elif acquisition == 'deterministic':
                        print('MC Consistency!')
                        posterior_dict_new, modality_dict_new, gt_labels_dict_new,_ = perform_MC_sampling(dropout_samples,save_path_dir,seed,epoch_count,batch_size,fraction,modalities,downstream_dataset,phases,acquisition,perturbation,mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,perturbed_dataloaders_list,models_list,mix_coefs,optimizer,device,inferences,acquired_indices,acquired_labels,input_perturbed,trial=trial,leads=leads,lambda1=lambda1)
                        perturbed_posterior_dict_new = None
            else:
                """ Function to Run Training """
                results_dictionary, outputs_list, labels_list, mix_coefs, modality_list, indices_list, task_names_list, scoring_function, hyperparam_dict = one_epoch(mixture,classification,criterion,criterion_single,weighted_sampling,phase,inference,dataloaders_list,models_list,mix_coefs,optimizer,device,hyperparam_dict=hyperparam_dict,trial=trial,epoch_count=epoch_count,lambda1=lambda1,save_path_dir=save_path_dir)

            """ Track Abstention Prob and Accuracy Per Sample """
            if 'train1' in phase:
                if epoch_count == 0:
                    abstention_matrix = []
                    hellinger_vector = []
                
                if trial == 'abstention_penalty':
                    gt_labels = np.concatenate(labels_list)
                    posterior_dists = np.concatenate(outputs_list)
                    
                    abstention_probs = expit(posterior_dists[:,-1])
                    #preds = obtain_prediction(posterior_dists,classification)
                    if classification == '2-way':
                        posterior_dists = expit(posterior_dists[:,:-1])
                        preds = np.where(posterior_dists>0.5,1,0)
                        gt_labels = np.expand_dims(gt_labels,1)
                        abstention_probs = np.expand_dims(abstention_probs,1)
                    elif classification is not None and classification != '2-way':
                        posterior_dists = softmax(posterior_dists[:,:-1],1)
                        preds = np.argmax(posterior_dists,1) #labels to assign to sample
                                    
                    #print(preds,gt_labels)
                    acc = np.where(preds==gt_labels,1,0)
                    #print(acc)
                    """ We Need a Fail Safe for No Zero Acc Samples or No One Acc Samples """
                    #print(abstention_probs.shape,(acc==0).shape)
                    incorrect_abstention_probs = abstention_probs[acc==0]
                    correct_abstention_probs = abstention_probs[acc==1]
                    gmm1 = GaussianMixture(1)
                    gmm2 = GaussianMixture(1)
                    
                    correct_probs = correct_abstention_probs.reshape(-1,1)
                    incorrect_probs = incorrect_abstention_probs.reshape(-1,1)
                    if len(correct_probs) > 2 and len(incorrect_probs) > 2:
                        gmm1.fit(correct_probs)
                        gmm2.fit(incorrect_probs)
                    
                        mean1,mean2 = gmm1.means_.item(), gmm2.means_.item()
                        cov1,cov2 = gmm1.covariances_.item(), gmm2.covariances_.item()
                        kld = -0.5 + np.log(np.sqrt(cov1)/np.sqrt(cov2)) + ((cov1 + (mean1 - mean2)**2)/(2*(cov2)))
                        hellinger = np.sqrt(1 - ( (np.sqrt((2*np.sqrt(cov1)*np.sqrt(cov2))/(cov1 + cov2))) * np.exp(((-1/4)*((mean1 - mean2)**2)/(cov1 + cov2))) ) ) 
                        print('KLD!, Hellinger!')
                        print(kld, hellinger)
                        abstention_threshold = {'gmm1':gmm1,'gmm2':gmm2}
                        
                        epoch_matrix = np.concatenate((np.expand_dims(acc,1),np.expand_dims(abstention_probs,1)),1)
                        abstention_matrix.append(epoch_matrix)
                        hellinger_vector.append(hellinger)
                        np.save(os.path.join(save_path_dir,'abstention'),abstention_matrix)
                        np.save(os.path.join(save_path_dir,'hellinger'),hellinger_vector)
                    else:
                        abstention_threshold = {'gmm1':None,'gmm2':None} #signals no GMM availability
                        hellinger = 0 #ensures dependence on oracle in scenario where all datapoints are misclassified   
                else:
                    abstention_threshold = {'gmm1':None,'gmm2':None} #filler
                    hellinger = 0 #ensures dependence on oracle in scenario where all datapoints are misclassified   
            
            """ Record Results """
            epoch_loss, epoch_acc, epoch_auroc = results_dictionary['epoch_loss'], results_dictionary['epoch_acc'], results_dictionary['epoch_auroc']

            """ Acquisition of New Datapoints Based on Acquisition Function """
            if 'train2' in phase and inference == True and 'time' not in metric: #remember train2 in this scenario will only happen if epoch is in acquisition_epochs
                #torch.save(posterior_dict_new,'posterior_dict')
                #torch.save(perturbed_posterior_dict_new,'perturbed_posterior_dict')
                """ Check if All Unlabelled Samples Have Been Acquired Already """
                if len(acquired_indices) != total_unlabelled_samples:
                    acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels,oracle_asks,proportion_wasted,_ = acquisition_function(downstream_dataset,save_path_dir,epoch_count,seed,metric,posterior_dict_new,modality_dict_new,gt_labels_dict_new,acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels,classification,acquisition_percent=acquisition_percent,perturbed_posterior_dict=perturbed_posterior_dict_new,trial=trial,abstention_threshold=abstention_threshold,hellinger=hellinger,hellinger_threshold=hellinger_threshold,oracle_asks=oracle_asks,noise_type=noise_type,noise_level=noise_level,nn_labels=nn_labels,tolerance=tolerance,proportion_wasted=proportion_wasted)
            elif 'train2' in phase and inference == True and 'time' in metric:
                acquisition_metric_dict,full_dict_for_saving = update_acquisition_dict(downstream_dataset,epoch_count,metric,classification,posterior_dict_new,acquisition_metric_dict,full_dict_for_saving,acquired_indices,trial,perturbed_posterior_dict_new)
                torch.save(full_dict_for_saving,os.path.join(save_path_dir,'area_under_aq_function'))
                if epoch_count in acquisition_epochs:
                    if len(acquired_indices) != total_unlabelled_samples:
                        acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels,oracle_asks,proportion_wasted,_ = acquisition_function(downstream_dataset,save_path_dir,epoch_count,seed,metric,posterior_dict_new,modality_dict_new,gt_labels_dict_new,acquired_indices,acquired_labels,acquired_modalities,acquired_gt_labels,classification,acquisition_percent=acquisition_percent,acquisition_metric_dict=acquisition_metric_dict,trial=trial,abstention_threshold=abstention_threshold,hellinger=hellinger,hellinger_threshold=hellinger_threshold,oracle_asks=oracle_asks,noise_type=noise_type,noise_level=noise_level,nn_labels=nn_labels,tolerance=tolerance,proportion_wasted=proportion_wasted)
                    #torch.save(acquisition_metric_dict,'acquisition_metric_dict')

            if inference == False:
                try:
                    print('%s Loss: %.4f. Acc: %.4f. AUROC: %.4f' % (phase,epoch_loss,epoch_acc,epoch_auroc))
                except:
                    print('%s Acc: %.4f. AUROC: %.4f' % (phase,epoch_acc,epoch_auroc))
                #print(scoring_function)
                if phase == 'val' and epoch_loss < best_loss or phase == 'test' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = [copy.deepcopy(model.state_dict()) for model in models_list]
                    """ Save Best Finetuned Weights """
                    if 'train1' in phases:
                        save_config_weights(save_path_dir,best_model_wts)
                    report, confusion = None, None
                    stop_counter = 0
                elif phase == 'val' and epoch_loss >= best_loss:
                    stop_counter += 1  
                    
                writer.add_scalar('%s_acc' % phase,epoch_acc,epoch_count)
                writer.add_scalar('%s_loss' % phase,epoch_loss,epoch_count)
                writer.add_scalar('%s_auc' % phase,epoch_auroc,epoch_count)
                acc_dict[phase].append(epoch_acc)
                loss_dict[phase].append(epoch_loss)
                auc_dict[phase].append(epoch_auroc)
            
        epoch_count += 1
        if 'train1' not in phases:
            break
            
    print('Best Val Loss: %.4f.' % best_loss)
    if 'train1' in phases:
        prefix = 'train_val'
        save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict)
        [model.load_state_dict(best_model_wt) for model,best_model_wt in zip(models_list,best_model_wts)]
    elif 'val' in phases:
        prefix = 'val'
        save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict)
    elif 'test' in phases:
        prefix = 'test'
        if input_perturbed == True: #save prefix for perturbation analysis 
            prefix = 'test_perturbed'
        save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict)        
    
    return models_list, report, confusion, epoch_loss, epoch_auroc

def make_saving_directory(phases,downstream_dataset,fraction,modalities,meta,acquisition_epochs,metric,seed,max_seed,acquisition='None',input_perturbed=False,perturbation='None',leads='None',trial='None',hyperparam='',tolerance='None',noise_type='None',hellinger_threshold='',noise_level=0):
    
    data_path = downstream_dataset
    
    #modality_path = ''
    #for mod in modalities:
    #    modality_path = mod + '+' + modality_path
    #modality_path = modality_path[:-1] #remove final + sign
    modality_path = modalities 
    
    if meta == True:
        meta_path = 'meta'
    else:
        meta_path = 'no_meta'
    
    fraction_path = 'fraction%s' % str(fraction)
        
    #print(len(acquisition_epochs),acquisition,input_perturbed,perturbation)
    if len(acquisition_epochs) > 0:
        active_path = 'active'
        metric_path = metric
        if acquisition == 'stochastic':
            if input_perturbed == False:
                acquisition_path = 'mc_dropout'
            elif input_perturbed == True:
                if perturbation == 'deterministic':
                    acquisition_path = 'balc'
        elif acquisition == 'deterministic':
            if input_perturbed == True:
                if perturbation == 'stochastic':
                    acquisition_path = 'mc_consistency'
    else:
        active_path = 'no_active'
        metric_path = ''
        acquisition_path = ''
    
    seed_path = 'seed%i' % int(seed)
    
    """ Change Base Path to Wherever You Want to Save """
    base_path = '/mnt/SecondaryHDD/Active Learning Results'
    #print(data_path,fraction_path,modality_path,meta_path,active_path,metric_path,acquisition_path)
    extra_path = os.path.join(data_path,fraction_path,modality_path,meta_path,active_path,metric_path,acquisition_path)
    
    if trial == 'None':
        trial_path = ''
        tolerance_path = ''
    else:
        trial_path = trial
        if trial == 'abstention_penalty':
            if tolerance != 'None':
                tolerance_path = 'tolerance_%s' % tolerance
            else:
                tolerance_path = ''
        else:
            tolerance_path = ''
    
    if noise_type == 'None':
        noise_type_path = ''
        noise_level_path = ''
    else:
        noise_type_path = 'label_noise_%s' % noise_type
        noise_level_path = 'noise_level_%s' % str(noise_level)
    
    if trial == 'abstention_penalty':
        if hellinger_threshold != 0.15:
            hellinger_path = 'hellinger_threshold_%.3f' % hellinger_threshold
        else:
            hellinger_path = ''
    else:
        hellinger_path = ''

    #print(trial_path,hyperparam,noise_type_path,noise_level_path,hellinger_path)
    if leads == 'None':
        save_path_dir = os.path.join(base_path,extra_path,trial_path,hyperparam,tolerance_path,noise_type_path,noise_level_path,hellinger_path,seed_path)
    else:
        leads_path = 'leads_%s' % leads
        save_path_dir = os.path.join(base_path,extra_path,leads_path,trial_path,hyperparam,tolerance_path,noise_type_path,noise_level_path,hellinger_path,seed_path)        

#    try:
#        os.chdir(save_path_dir)
#        if 'finetuned_weight' in os.listdir():
#            seed_path = 'seed%i' % (int(seed) + 1)
#            extra_path = os.path.join(data_path,fraction_path,modality_path,meta_path,active_path,metric_path,acquisition_path,seed_path)
#            save_path_dir = os.path.join(base_path,extra_path)
#
#    except:
#        os.makedirs(save_path_dir)
    if 'train' in phases:
        save_path_dir, seed = make_dir(save_path_dir,base_path,extra_path,leads,trial,seed,max_seed,hyperparam,tolerance_path,noise_type_path,noise_level_path,hellinger_path)
    
    return save_path_dir, seed

#%%
def make_dir(save_path_dir,base_path,extra_path,leads,trial,seed,max_seed,hyperparam,tolerance_path,noise_type_path,noise_level_path,hellinger_path):
    """ Recursive Function to Make Sure I do Not Overwrite Previous Seeds """
    try:
        os.chdir(save_path_dir)
        if 'train_val_auc' in os.listdir():
            if int(seed) < max_seed-1:
                print('Skipping Seed!')
                seed = int(seed) + 1
                seed_path = 'seed%i' % seed
                if trial == 'None':
                    trial_path = ''
                else:
                    trial_path = trial

                if leads == 'None':
                    save_path_dir = os.path.join(base_path,extra_path,trial_path,hyperparam,tolerance_path,noise_type_path,noise_level_path,hellinger_path,seed_path)
                else:
                    leads_path = 'leads_%s' % leads
                    save_path_dir = os.path.join(base_path,extra_path,leads_path,trial_path,hyperparam,tolerance_path,noise_type_path,noise_level_path,hellinger_path,seed_path)        
                print(save_path_dir)
                save_path_dir, seed = make_dir(save_path_dir,base_path,extra_path,leads,trial,seed,max_seed,hyperparam,tolerance_path,noise_type_path,noise_level_path,hellinger_path)
            else:
                save_path_dir = 'do not train'
    except:
        os.makedirs(save_path_dir)
    
    if int(seed) == max_seed:
        seed = 0
    
    return save_path_dir, int(seed)

def save_config_weights(save_path_dir,best_model_weights):
    torch.save(best_model_weights,os.path.join(save_path_dir,'finetuned_weight'))

def save_statistics(save_path_dir,prefix,acc_dict,loss_dict,auc_dict):
    torch.save(acc_dict,os.path.join(save_path_dir,'%s_acc' % prefix))
    torch.save(loss_dict,os.path.join(save_path_dir,'%s_loss' % prefix))
    torch.save(auc_dict,os.path.join(save_path_dir,'%s_auc' % prefix))
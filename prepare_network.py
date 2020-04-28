#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:09:14 2020

@author: scro3517
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

c1 = 1 #b/c single time-series
c2 = 4 #4
c3 = 16 #4
c4 = 32 #4
k=7 #kernel size 
s=3 #stride 
#num_classes = 3

class cnn_network_time(nn.Module):
    
    """ CNN Implemented in Original Paper - Supposedly Simple but Powerful """
    
    def __init__(self,dropout_type,p1,p2,p3,classification,heads='single'):
        super(cnn_network_time,self).__init__()
        
        if classification is not None and classification != '2-way':
            num_classes = int(classification.split('-')[0])
        elif classification == '2-way':
            num_classes = 1
        
        embedding_dim = 100 #100
        
        #self.conv1 = nn.Conv2d(c1,c2,k,s)
        self.conv1 = nn.Conv1d(c1,c2,k,s)
        self.batchnorm1 = nn.BatchNorm1d(c2)
        #self.conv2 = nn.Conv2d(c2,c3,k,s)
        self.conv2 = nn.Conv1d(c2,c3,k,s)
        self.batchnorm2 = nn.BatchNorm1d(c3)
        #self.conv3 = nn.Conv2d(c3,c4,k,s)
        self.conv3 = nn.Conv1d(c3,c4,k,s)
        self.batchnorm3 = nn.BatchNorm1d(c4)
        self.linear1 = nn.Linear(c4*10,embedding_dim)
        self.linear2 = nn.Linear(embedding_dim,num_classes)
        
        self.oracle_head = nn.Linear(embedding_dim,1) #I may have to comment out when performing inference for ALPS
        self.heads = heads

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool1d(2)
        #self.fracmaxpool = nn.FractionalMaxPool2d(2,output_ratio=0.50) #kernel size, output size relative to input size
        
        if dropout_type == 'drop1d':
            self.dropout1 = nn.Dropout(p=p1) #0.2 drops pixels following a Bernoulli
            self.dropout2 = nn.Dropout(p=p2) #0.2
            self.dropout3 = nn.Dropout(p=p3)
        elif dropout_type == 'drop2d':
            self.dropout1 = nn.Dropout2d(p=p1) #drops channels following a Bernoulli
            self.dropout2 = nn.Dropout2d(p=p2)
            self.dropout3 = nn.Dropout2d(p=p3)
        
        #self.alphadrop1 = nn.AlphaDropout(p=0.1) #used primarily with selu activation
        
    def forward(self,x):
        x = self.dropout1(self.maxpool(self.relu(self.batchnorm1(self.conv1(x)))))
        x = self.dropout2(self.maxpool(self.relu(self.batchnorm2(self.conv2(x)))))
        x = self.dropout3(self.maxpool(self.relu(self.batchnorm3(self.conv3(x)))))
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        x = self.relu(self.linear1(x))
        out = self.linear2(x)
        if self.heads == 'multi':
            p = self.oracle_head(x)
            return (out,p)
        else:
            return out

#%%
class cnn_network_image(nn.Module):
    def __init__(self,dropout_type,p1,p2,p3,classification,heads='single'):
        super(cnn_network_image, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.dropout1 = nn.Dropout(p=p1) #0.2 drops pixels following a Bernoulli
        self.dropout2 = nn.Dropout(p=p2) #0.2
        #self.dropout3 = nn.Dropout(p=p3)
        
        self.oracle_head = nn.Linear(84,1) #I may have to comment out when performing inference for ALPS
        self.heads = heads

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        out = self.fc3(x)
        if self.heads == 'multi':
            p = self.oracle_head(x)
            return (out,p)
        else:
            return out     
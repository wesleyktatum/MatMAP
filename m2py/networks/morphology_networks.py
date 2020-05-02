"""
This module contains neural networks for predicting device performance,
based on tabular data and m2py labels
"""

import torch
import torch.nn as nn


class OPV_df_NN(nn.Module):
    """
    expects tabular data predicting 4 OPV device metrics: PCE, Voc, Jsc, and FF
    """
    
    def __init__(self, in_dims, out_dims):
        super(OPV_df_NN, self).__init__()
        
        #emedding layer
        self.em_layer = nn.Linear(in_dims, out_dims)
        
        #hidden layers
        self.h_layer1 = nn.Linear(out_dims, 32)
        self.h_layer2 = nn.Linear(32, 16)
        self.h_layer3 = nn.Linear(16, 8)
        
        #output layers
        self.PCE_branch = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 16),
            nn.Dropout(p = 0.3),
            nn.Softplus(),
            nn.Linear(16, 1)
        )
        self.Voc_branch = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 16),
            nn.Dropout(p = 0.3),
            nn.Softplus(),
            nn.Linear(16, 1)
        )
        self.Jsc_branch = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 16),
            nn.Dropout(p = 0.3),
            nn.Softplus(),
            nn.Linear(16, 1)
        )
        self.FF_branch = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 16),
            nn.Dropout(p = 0.3),
            nn.Softplus(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        #data enters embedding layer
        out = self.em_layer(x)
        
        #embedded data is passed to hidden layers
        out = self.h_layer1(out)
        out = self.h_layer2(out)
        out = self.h_layer3(out)
        
        #embedded data is passed to output layer
        PCE_out = self.PCE_branch(out)
        Voc_out = self.Voc_branch(out)
        Jsc_out = self.Jsc_branch(out)
        FF_out = self.FF_branch(out)
        
        return PCE_out, Voc_out, Jsc_out, FF_out


class OPV_m2py_NN(nn.Module):
    """
    expects m2py labels of size 256x256x2
    """
    
    def __init__(self, im_z):
        super(OPV_m2py_NN, self).__init__()
        
#         fc_nodes = int((im_x/4) * (im_y/4) * 64) # 4 because there were 2 maxpool layers with 2x2 Kernals,
                                                 # reducing the overall channel width & height by half each time
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(im_z, 32, kernel_size = 8, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Dropout(),               #helps avoid over-fitting
            nn.Linear(6668032, 10000),
            nn.Linear(10000, 5000),
            nn.Linear(5000, 1000),
            nn.Linear(1000, 100)
        )
        
        self.out_layer = nn.Linear(100, 2) #predicting anneal time and temp
            
    def forward(self, x_im):
        im_out = self.layer1(x_im)
        im_out = self.layer2(im_out)
        
        im_out = im_out.view(-1,1) #reshape output for linear layers
        
        im_out = self.layer3(im_out)
        im_train_out = self.out_layer(im_out)
        
        return im_out, im_train_out
    
    
    #define the neural network
class OFET_df_NN(nn.Module):
    
    def __init__(self, in_dims, out_dims):
        super(OFET_df_NN, self).__init__()
        
        #emedding layer
        self.em_layer = nn.Linear(in_dims, out_dims)
        
        #hidden layers
        self.h_layer1 = nn.Linear(out_dims, 32)
        self.h_layer2 = nn.Linear(32, 16)
        self.h_layer3 = nn.Linear(16, 8)
        
        #output layers
        self.mu_branch = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 16),
            nn.Dropout(p = 0.3),
            nn.Softplus(),
            nn.Linear(16, 1)
        )
        self.r_branch = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 16),
            nn.Dropout(p = 0.3),
            nn.Softplus(),
            nn.Linear(16, 1)
        )
        self.on_off_branch = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 16),
            nn.Dropout(p = 0.3),
            nn.Softplus(),
            nn.Linear(16, 1)
        )
        self.vt_branch = nn.Sequential(
            nn.Dropout(p = 0.3),
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 16),
            nn.Dropout(p = 0.3),
            nn.Softplus(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        #data enters embedding layer
        out = self.em_layer(x)
        
        #embedded data is passed to hidden layers
        out = self.h_layer1(out)
        out = self.h_layer2(out)
        out = self.h_layer3(out)
        
        #embedded data is passed to output layer
        mu_out = self.mu_branch(out)
        r_out = self.r_branch(out)
        on_off_out = self.on_off_branch(out)
        vt_out = self.vt_branch(out)
        
        return mu_out, r_out, on_off_out, vt_out
    
    
    
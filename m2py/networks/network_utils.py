"""
This modules contains utility functions for data manipulation and plotting of
results and data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch


#######################################################
#                  Data Utilities    
#######################################################

def load_trained_model(previous_model, model, optimizer):
    
    checkpoint = torch.load(previous_model)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
        
    return model, optimizer


def save_trained_model(save_path, epoch, model, optimizer, train_loss, test_loss):
    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
#         'train_losses': train_loss
#         'test_losses': [pce_test_loss, voc_test_loss,
#                        jsc_test_loss, ff_test_loss]
        'optimizer': optimizer.state_dict()
        }
        
    torch.save(save_dict, save_path)
    return


def df_MinMax_normalize(dataframe):
    
    df = dataframe
    
    normed_df = pd.DataFrame()

    df_norm_key = {}

    for colname, coldata in df.iteritems():
        max_val = coldata.max()
        min_val = coldata.min()

        df_norm_key[colname] = [min_val, max_val]

        normed_col = (coldata - min_val) / (max_val - min_val)
        normed_df[colname] = normed_col
        
    return normed_df, df_norm_key 


def df_MinMax_denormalize(normed_df, norm_key):
    
    denormed_df = pd.DataFrame()
    
    for colname, coldata in normed_df.iteritems():
        mn = norm_key[colname][0]
        mx = norm_key[colname][1]
        
        denormed_col = (coldata * (mx - mn)) + mn
        
        denormed_df[colname] = denormed_col
        
    return denormed_df


def df_Gaussian_normalize(dataframe):
    
    df = dataframe
    normed_df = pd.DataFrame()
    norm_key = {}
    
    for colname, coldata in df.iteritems():
        stdev = coldata.std()
        mean = coldata.mean()
        
        normed_col = (coldata - mean) / stdev
        normed_df[colname] = normed_col
        
        norm_key[colname] = [mean, stdev]
        
    return normed_df, norm_key


#######################################################
#                  Network Model Utilities
#######################################################

def init_weights(model):
    if type(model) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)
        
#     if type(model) == nn.BatchNorm1d:
#         model.reset_parameters()

    

#######################################################
#                  Plotting Utilities
#######################################################

def plot_OPV_df_loss(epochs, train_epoch_losses, test_epoch_losses,
                     pce_train_epoch_losses, pce_test_epoch_losses,
                     voc_train_epoch_losses, voc_test_epoch_losses,
                     jsc_train_epoch_losses, jsc_test_epoch_losses,
                     ff_train_epoch_losses, ff_test_epoch_losses):
    
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    plt.plot(epochs, train_epoch_losses, c = 'k', label = 'training error')
    plt.plot(epochs, test_epoch_losses, c = 'r', label = 'testing error')
    plt.legend(loc = 'upper right')
    plt.title("Total Training & Testing Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total MSE Loss')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (8,6))

    plt.plot(epochs[::], pce_train_epoch_losses[::], c = 'k', label = 'pce training')
    plt.plot(epochs[::], pce_test_epoch_losses[::], '-.', c = 'k', label = 'pce testing')

    plt.plot(epochs[::], voc_train_epoch_losses[::], c = 'r', label = 'voc training')
    plt.plot(epochs[::], voc_test_epoch_losses[::], '-.', c = 'r', label = 'voc testing')

    plt.plot(epochs[::], jsc_train_epoch_losses[::], c = 'g', label = 'jsc training')
    plt.plot(epochs[::], jsc_test_epoch_losses[::], '-.', c = 'g', label = 'jsc testing') 
    
    plt.plot(epochs[::], ff_train_epoch_losses[::], c = 'b', label = 'ff training') 
    plt.plot(epochs[::], ff_test_epoch_losses[::], '-.', c = 'b', label = 'ff testing') 

    plt.legend(loc = 'upper right')
    plt.title("Branch Training & Testing Error")
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE')
    plt.show()
    
    return

def plot_OPV_df_accuracies(epochs, pce_test_epoch_accuracies, voc_test_epoch_accuracies, 
                           jsc_test_epoch_accuracies, ff_test_epoch_accuracies):
    
    fig, ax = plt.subplots(figsize = (8,6))
    # plt.plot(epochs, train_epoch_accuracy, c = 'k', label = 'training accuracy')
    plt.plot(epochs, pce_test_epoch_accuracies, c = 'k', label = 'pce MAPE')
    plt.plot(epochs, voc_test_epoch_accuracies, c = 'r', label = 'voc MAPE')
    plt.plot(epochs, jsc_test_epoch_accuracies, c = 'g', label = 'jsc MAPE')
    plt.plot(epochs, ff_test_epoch_accuracies, c = 'b', label = 'ff MAPE')
    plt.legend(loc = 'upper right')
    plt.title("Branch Testing Accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Percent Error')
    plt.show()
    
    return

def plot_OPV_parity(pce_labels, PCE_out, voc_labels, Voc_out,
                    jsc_labels, Jsc_out, ff_labels, FF_out):
    
    xlin = ylin = np.arange(-10, 10, 1)

    r2 = r2_score(pce_labels, PCE_out)
    fig, ax = plt.subplots(figsize = (8,6))
    plt.scatter(PCE_out, pce_labels)
    plt.plot(xlin, ylin, c = 'k')
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4))
    ax.set_xlim(min(pce_labels.min(), PCE_out.min()), max(pce_labels.max(), PCE_out.max()))
    ax.set_ylim(min(pce_labels.min(), PCE_out.min()), max(pce_labels.max(), PCE_out.max()))
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    plt.title('PCE Parity')
    plt.show()

    r2 = r2_score(voc_labels, Voc_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4))
    plt.scatter(Voc_out, voc_labels)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(min(voc_labels.min(), Voc_out.min()), max(voc_labels.max(), Voc_out.max()))
    ax.set_ylim(min(voc_labels.min(), Voc_out.min()), max(voc_labels.max(), Voc_out.max()))
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    plt.title('Voc Parity')
    plt.show()

    r2 = r2_score(jsc_labels, Jsc_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4))
    plt.scatter(Jsc_out, jsc_labels)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(min(jsc_labels.min(), Jsc_out.min()), max(jsc_labels.max(), Jsc_out.max()))
    ax.set_ylim(min(jsc_labels.min(), Jsc_out.min()), max(jsc_labels.max(), Jsc_out.max()))
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    plt.title('Jsc Parity')
    plt.show()

    r2 = r2_score(ff_labels, FF_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4))
    plt.scatter(FF_out, ff_labels)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(min(ff_labels.min(), FF_out.min()), max(ff_labels.max(), FF_out.max()))
    ax.set_ylim(min(ff_labels.min(), FF_out.min()), max(ff_labels.max(), FF_out.max()))
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    plt.title('FF Parity')
    plt.show()
    
    
def plot_OFET_df_loss(epochs, train_epoch_losses, test_epoch_losses,
                      mu_train_epoch_losses, mu_test_epoch_losses,
                      r_train_epoch_losses, r_test_epoch_losses,
                      on_off_train_epoch_losses, on_off_test_epoch_losses,
                      vt_train_epoch_losses, vt_test_epoch_losses):
    
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    plt.plot(epochs[::], train_epoch_losses[::], c = 'k', label = 'training error')
    plt.plot(epochs[::], test_epoch_losses[::], c = 'r', label = 'testing error')
    plt.legend(loc = 'upper right')
    plt.title("Total Training & Testing Error")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total MSE Loss')
    plt.show()
    
    fig, ax = plt.subplots(figsize = (8,6))

    plt.plot(epochs[::], mu_train_epoch_losses[::], c = 'k', label = 'mu training')
    plt.plot(epochs[::], mu_test_epoch_losses[::], '-.', c = 'k', label = 'mu testing')

    plt.plot(epochs[::], r_train_epoch_losses[::], c = 'r', label = 'r training')
    plt.plot(epochs[::], r_test_epoch_losses[::], '-.', c = 'r', label = 'r testing')

    plt.plot(epochs[::], on_off_train_epoch_losses[::], c = 'g', label = 'on_off training')
    plt.plot(epochs[::], on_off_test_epoch_losses[::], '-.', c = 'g', label = 'on_off testing') 
    
    plt.plot(epochs[::], vt_train_epoch_losses[::], c = 'b', label = 'vt training') 
    plt.plot(epochs[::], vt_test_epoch_losses[::], '-.', c = 'b', label = 'vt testing') 

    plt.legend(loc = 'upper right')
    plt.title("Branch Training & Testing Error")
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE')
    plt.show()
    
    return


def plot_OFET_df_accuracies(epochs, mu_test_epoch_accuracies, r_test_epoch_accuracies, 
                            on_off_test_epoch_accuracies, vt_test_epoch_accuracies):
    
    fig, ax = plt.subplots(figsize = (8,6))
    # plt.plot(epochs, train_epoch_accuracy, c = 'k', label = 'training accuracy')
    plt.plot(epochs, mu_test_epoch_accuracies, c = 'k', label = 'mu MAPE')
    plt.plot(epochs, r_test_epoch_accuracies, c = 'r', label = 'r MAPE')
    plt.plot(epochs, on_off_test_epoch_accuracies, c = 'g', label = 'on_off MAPE')
    plt.plot(epochs, vt_test_epoch_accuracies, c = 'b', label = 'vt MAPE')
    plt.legend(loc = 'upper right')
    plt.title("Branch Testing Accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Percent Error')
    plt.show()
    
    return


def plot_OFET_parity(mu_labels, mu_out, r_labels, r_out,
                     on_off_labels, on_off_out, vt_labels, vt_out):
    
    xlin = ylin = np.arange(-20, 20, 1)

    r2 = r2_score(mu_labels, mu_out)
    fig, ax = plt.subplots(figsize = (8,6))
    plt.scatter(mu_labels, mu_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('mu Parity')
    plt.show()

    r2 = r2_score(r_labels, r_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4))
    plt.scatter(r_labels, r_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('r Parity')
    plt.show()

    r2 = r2_score(on_off_labels, on_off_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4))
    plt.scatter(on_off_labels, on_off_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('on_off Parity')
    plt.show()

    r2 = r2_score(vt_labels, vt_out)
    fig, ax = plt.subplots(figsize = (8,6))
    ax.annotate(f"$R^{2}$ = {r2:.3f}", xy = (0.2, 0.4))
    plt.scatter(vt_labels, vt_out)
    plt.plot(xlin, ylin, c = 'k')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Predictions")
    ax.set_xlabel("Ground Truth")
    plt.title('Vt Parity')
    plt.show()
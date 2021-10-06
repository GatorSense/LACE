#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:56:56 2021

@author: cmccurley
"""

import argparse

import os
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from barbar import Bar
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.preprocessing import normalize
import pdb

# from CM_Parameters import Parameters
from Demo_Parameters import Parameters
from Utils.RBFHistogramPooling import HistogramLayer
from Prepare_Data import Prepare_DataLoaders
from Utils.Network_functions import initialize_model
from Utils.Generate_Learning_Curves import Plot_Learning_Curves

##Turn off plotting
plt.ioff()

###Activate LaTex rendering
#rc('text', usetex=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Losses and Baseline experiments for dataset')
    parser.add_argument('--save_results', type=int, default=True,
                        help='Save results of experiments(default: True')
    parser.add_argument('--folder', type=str, default='Results/',
                        help='Location to save models')
    parser.add_argument('--histogram', type=bool, default=False,
                        help='Flag to use histogram model or baseline global average pooling (GAP)')
    parser.add_argument('--data_selection', type=int, default=8,
                        help='Dataset selection:  1: DTD, 2: GTOS-mobile, 3: MINC_2500, 4: KTH_TIPS, 5: MNIST, 6:FashionMNIST, 7:SVHN, 8:CIFAR10')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', type=bool, default=True,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', type=bool, default=True,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--weights', type=list, default=[1],
                        help=' Set weights for objective term: value(s) should be between 0 and 1. (default: [.25, .5, .75, 1]')
    parser.add_argument('--embed_dim', type=list, default=[512],
                        help=' Embedding dimension of encoder. (default: [3], will also run full dimension size)')
    parser.add_argument('--regularization_method', type=list, default=['LACES'], 
                        help='Feature regularization approach to use (default: all methods)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()
    return args

def visulaize_embeddings(background_means, background_covs, target_signatures, 
                         model_ft, dataloaders_dict, regularization_method, whiten_path, non_whiten_path,
                         split):
    """
    ******************************************************************
        *  Func:    visulaize_embeddings()
        *  Desc:    E
        *  Inputs:    
        *           background_means -
        *               Cur
        *
        *           background_covs -
        *               To
        *
        *           target_signatures -
        *               Ne
        *
        *           model_ft -
        *               Ins
        *
        *           data_loader -
        *               String 
        *
        *
        *  Outputs:   
        *           loss
        *     
    ******************************************************************
    """
    
    ## Embed training data
    print('Embedding {} data...\n'.format(split))
    for idx, (inputs, labels, index) in enumerate(Bar(dataloaders_dict[split])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        index = index.to(device)
        background_means = background_means.to(device)
        background_covs = background_covs.to(device)
        target_signatures = target_signatures.to(device)
            
        # Get model outputs and calculate loss
        outputs, embedding, loss_embedding = model_ft(inputs,labels)
        
        if (idx == 0):
            ## Initialize data structures
            Z_train = embedding
            Y_train = labels
            
        else:
            ## Add data
            Z_train = torch.cat((Z_train,embedding),dim=0)
            Y_train = torch.cat((Y_train,labels))
    
    
    ## Visualize t-SNE projection considering each class as the target
            
    ##################### Create directory to save figures ####################
    sub_whiten_path = whiten_path + '/' + split
    try:
        os.mkdir(sub_whiten_path)
    except OSError:
        shutil.rmtree(sub_whiten_path, ignore_errors=True)
        os.mkdir(sub_whiten_path)
    
    sub_non_whiten_path = non_whiten_path + '/' + split
    try:
        os.mkdir(sub_non_whiten_path)
    except OSError:
        shutil.rmtree(sub_non_whiten_path, ignore_errors=True)
        os.mkdir(sub_non_whiten_path)
    
    for label_idx in torch.unique(Y_train):
       
        # Z_tsne_train = TSNE(n_components=2, metric='euclidean', random_state=0,init='pca').fit_transform(Z_train.detach().cpu())
        # # pdb.set_trace()
        label_idx = label_idx.detach().cpu()
        # plt.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],0],Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],1],c='orange')
        # plt.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],0],Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],1],c='blue')
        # plt.title('Euclidean, no Whitening for Class {}'.format(label_idx.detach().cpu().numpy()))
        # savename = sub_non_whiten_path + '/Euclidean_class_' + str(label_idx.detach().cpu().numpy()) + '.png'
        # plt.savefig(savename)
        # plt.close()
        # plt.show()
        
        Z_tsne_train = TSNE(n_components=2, metric='cosine', random_state=0, init='pca').fit_transform(Z_train.detach().cpu())
        plt.figure()
        plt.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],0],Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],1],c='orange')
        plt.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],0],Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],1],c='blue')
        plt.title('Cosine, no Whitening for Class {}'.format(label_idx.detach().cpu().numpy()))
        savename = sub_non_whiten_path + '/Cosine_class_' + str(label_idx.detach().cpu().numpy()) + '.png'
        plt.savefig(savename)
        plt.close()
        # plt.show()
        
        # #Visualize on sphere
        tsne = TSNE(n_components=3, metric='cosine',init='pca', random_state=0)
        Z_tsne_train = tsne.fit_transform(Z_train.detach().cpu())
        
        #L2 normalize 
        Z_tsne_train = normalize(Z_tsne_train)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a sphere
        r = 1
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
        x = r*sin(phi)*cos(theta)
        y = r*sin(phi)*sin(theta)
        z = r*cos(phi)
        ax.plot_surface(
            x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
        ax.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],0],
                   Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],1],
                   Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],2],
                   c='orange',s=20)
        ax.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],0],
                   Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],1],
                   Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],2],
                   c='blue', s=20)
    
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        plt.title('Cosine, no Whitening for Class {}'.format(label_idx.detach().cpu().numpy()))
        savename = sub_non_whiten_path + '/3D_Cosine_class_' + str(label_idx.detach().cpu().numpy()) + '.png'
        plt.savefig(savename)
        pdb.set_trace()
        plt.close()
            # ax.grid(False)
        
        
        ## Whiten data
        try: #Multiple background statistics
            Z_train = Z_train- background_means[label_idx,:]
            U, D, _ = torch.svd(torch.mm(background_covs[label_idx,:,:],
                                          background_covs[label_idx,:,:].T))
        except:
            Z_train = Z_train- background_means
            U, D, _ = torch.svd(torch.mm(background_covs,background_covs.T))
            
        D_inv_half = torch.diag(torch.pow(D,-0.5))
        E = torch.matmul(D_inv_half,U.T)
        
        Z_train = torch.matmul(Z_train,E)
    
        # Z_tsne_train = TSNE(n_components=2, metric='euclidean', random_state=0,init='pca').fit_transform(Z_train.detach().cpu())
        # plt.figure()
        # plt.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],0],Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],1],c='orange')
        # plt.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],0],Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],1],c='blue')
        # plt.title('Euclidean, Whitening for Class {}'.format(label_idx.detach().cpu().numpy()))
        # savename = sub_whiten_path + '/Euclidean_class_' + str(label_idx.detach().cpu().numpy()) + '.png'
        # plt.savefig(savename)
        # plt.close()
        # plt.show()
        
        Z_tsne_train = TSNE(n_components=2, metric='cosine',random_state=0, init='pca').fit_transform(Z_train.detach().cpu())
        plt.figure()
        plt.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],0],Z_tsne_train[np.where(Y_train.detach().cpu()==label_idx)[0],1],c='orange')
        plt.scatter(Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],0],Z_tsne_train[np.where(Y_train.detach().cpu()!=label_idx)[0],1],c='blue')
        plt.title('Cosine, Whitening for Class {}'.format(label_idx.detach().cpu().numpy()))
        savename = sub_whiten_path + '/Cosine_class_' + str(label_idx.detach().cpu().numpy()) + '.png'
        plt.savefig(savename)
        plt.close()
        # plt.show()
    
        # print('here')
        # pdb.set_trace()
    
    
    ## TSNE Cosine
    
    ##TSNE Euclidean
    
def visulaize_angle_dist_non_ace(model_ft, dataloaders_dict, split, regularization_method, cos_path):
    """
    ******************************************************************
        *  Func:    visulaize_angle_dist_non_ace()
        *
        *  Desc:    Generates histograms of cosine similarities between
        *           samples and corresponding target represenatives, 
        *           computed in the embedding space.
        *
        *  Inputs:    
        *
        *           model_ft:
        *               Neural network embedding model
        *
        *           dataloaders_dict:
        *               Dictionary containing 'train', 'val' and 'test'
        *               Pytorch data loaders
        *
        *           split:
        *               String denoting the current dataset to consider.
        *               (Options: 'train', 'val', 'test')
        *
        *           regularization_method:
        *               String denoting the regularization used in training
        *               the model.
        *               (E.g.: 'AMC', 'LACES', 'softmax')
        *
        *           cos_path:
        *               String denoting the base path to save the figures.
        *     
    ******************************************************************
    """
    
    ##### Compute class representatives from means of training data ###########
    ############# and center the data for cosine computation ##################
    
    ## Embed training data
    print('\nComputing class representatives for {} data...\n'.format(split))
    for idx, (inputs, labels, index) in enumerate(Bar(dataloaders_dict[split])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        index = index.to(device)
        
        # Get model outputs and calculate loss
        outputs, embedding, loss_embedding = model_ft(inputs,labels)
        
        if (idx == 0):
            ## Initialize data structures
            Z = embedding
            Y = labels
        
        else:
            ## Add data
            Z = torch.cat((Z,embedding),dim=0)
            Y = torch.cat((Y,labels))
    
    ## Zero center data        
    total_mean = torch.mean(Z,dim=0)
    Z = Z - total_mean
           
    ## Compute class means and set as target signatures
    for idx, label_idx in enumerate(torch.unique(Y)):
        class_mean = torch.mean(Z[np.where(Y.detach().cpu().numpy()==label_idx.detach().cpu().numpy())[0],:],dim=0)
        class_mean = torch.unsqueeze(class_mean,dim=0)
        
        if (idx == 0):
            ## Initialize data structures
            target_signatures = class_mean
            
        else:
            ## Add data
            target_signatures = torch.cat((target_signatures,class_mean),dim=0)
    
    ########################### Embed data ####################################
    print('\n Embedding {} data...\n'.split(split))
    for idx, (inputs, labels, index) in enumerate(Bar(dataloaders_dict[split])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        index = index.to(device)
            
        # Get model outputs and calculate loss
        outputs, embedding, loss_embedding = model_ft(inputs,labels)
        
        if (idx == 0):
            ## Initialize data structures
            Z_train = embedding
            Y_train = labels
            
        else:
            ## Add data
            Z_train = torch.cat((Z_train,embedding),dim=0)
            Y_train = torch.cat((Y_train,labels))
    
    ## Zero center data
    Z_train = Z_train - total_mean
    
    ##################### Create directory to save figures ####################
    sub_cos_path = cos_path + '/' + split
                        
    try:
        os.mkdir(sub_cos_path)
    except OSError:
        shutil.rmtree(sub_cos_path, ignore_errors=True)
        os.mkdir(sub_cos_path)
    
    ################# Visualize distributions of angles to target #############
    
    for label_idx in torch.unique(Y_train):
        
        ######################## Cosine Similarity ############################
        
        ## Cosine similarity histograms 
        Z_train_norm = F.normalize(Z_train,p=2,dim=1)
        S_norm = F.normalize(target_signatures,p=2,dim=1)
        cosine_scores = torch.matmul(Z_train_norm,S_norm[label_idx,:])
#        cosine_angles = torch.acos(cosine_scores)*(180/3.14)
        
        ## Plot histogram of angles
        target_idx = np.where(Y_train.detach().cpu()==label_idx.detach().cpu())[0]
        bg_idx = np.where(Y_train.detach().cpu()!=label_idx.detach().cpu())[0]
        cosine_angles = cosine_scores.detach().cpu().numpy() + 1
        cosine_angles_target = cosine_angles[target_idx]
        cosine_angles_bg = cosine_angles[bg_idx]
        
        min_target_sim = np.min(cosine_angles_target)
        max_bg_sim = np.max(cosine_angles_bg)
        min_target_sim = np.round(min_target_sim,decimals=3)
        max_bg_sim = np.round(max_bg_sim,decimals=3)
        
        ## Make histograms
        plt.figure()
        legend0 = 'Target Class ' + str(label_idx.detach().cpu().numpy())
        legend1 = 'Background Class ' + str(label_idx.detach().cpu().numpy())
        plt.hist(cosine_angles_bg,bins=np.arange(-0.01,2.01,0.02), density=True, color='blue', label=legend1, stacked=True, alpha=0.7, edgecolor='black', linewidth=1.2)
        plt.hist(cosine_angles_target,bins=np.arange(-0.01,2.01,0.02), density=True, color='darkorange', label=legend0, stacked=True, alpha=0.6, edgecolor='black', linewidth=1.2)
        title = 'Cosine Similarity to Class ' + str(label_idx.detach().cpu().numpy()) +' Mean'
        plt.gca().set(title=title, xlabel='Cosine Similarity + 1')
        plt.legend(loc='upper left')
        plt.figtext(0.15,0.7,'min target sim: %.3f' % min_target_sim)
        plt.figtext(0.15,0.66,'max bg sim: %.3f' % max_bg_sim)
        savename = sub_cos_path + '/class_' + str(label_idx.detach().cpu().numpy()) + '.png'
        plt.savefig(savename)
        # plt.show()
    
    return

def visulaize_angle_dist_ace(background_means, background_covs, target_signatures, model_ft, dataloaders_dict, split, regularization_method, cos_path, ace_path):
    
    """
    ******************************************************************
        *  Func:    visulaize_angle_dist_ace()
        *
        *  Desc:    Generates histograms of cosine similarities between
        *           samples and corresponding target represenatives, 
        *           computed in the embedding space. Also computes the ACE
        *           similarities in the whitened space (whitened by learned
        *           global background mean and covariance).
        *
        *  Inputs:    
        *
        *           background_means:
        *               [1 x d] tensor representing the learned background mean 
        *
        *           background_covs:
        *               [d x d] tensor representing the learned background covariance
        *
        *           target_signatures:
        *               [C x d] tensor of learned class representatives
        *
        *           model_ft:
        *               Neural network embedding model
        *
        *           dataloaders_dict:
        *               Dictionary containing 'train', 'val' and 'test'
        *               Pytorch data loaders
        *
        *           split:
        *               String denoting the current dataset to consider.
        *               (Options: 'train', 'val', 'test')
        *
        *           regularization_method:
        *               String denoting the regularization used in training
        *               the model.
        *               (E.g.: 'AMC', 'LACES', 'softmax')
        *
        *           cos_path:
        *               String denoting the base path to save the cosine figures.
        *     
        *           ace_path:
        *               String denoting the base path to save the ACE figures.
        *    
    ******************************************************************
    """
    
    ########################### Embed data ####################################
    print('\nEmbedding data...\n')
    for idx, (inputs, labels, index) in enumerate(Bar(dataloaders_dict[split])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        index = index.to(device)
        background_means = background_means.to(device)
        background_covs = background_covs.to(device)
        target_signatures = target_signatures.to(device)
            
        # Get model outputs and calculate loss
        outputs, embedding, loss_embedding = model_ft(inputs,labels)
        
        if (idx == 0):
            ## Initialize data structures
            Z_train = embedding
            Y_train = labels
            
        else:
            ## Add data
            Z_train = torch.cat((Z_train,embedding),dim=0)
            Y_train = torch.cat((Y_train,labels))
    

    ##################### Create directory to save figures ####################
    sub_cos_path = cos_path + '/' + split
    try:
        os.mkdir(sub_cos_path)
    except OSError:
        shutil.rmtree(sub_cos_path, ignore_errors=True)
        os.mkdir(sub_cos_path)
    
    sub_ace_path = ace_path + '/' + split
    try:
        os.mkdir(sub_ace_path)
    except OSError:
        shutil.rmtree(sub_ace_path, ignore_errors=True)
        os.mkdir(sub_ace_path)
    
    ################ Visualize distributions of angles to target ##############
    
    for label_idx in torch.unique(Y_train):
        
        ######################## Cosine Similarity ############################
        
        ## Cosine similarity histograms 
        Z_train_norm = F.normalize(Z_train,p=2,dim=1)
        S_norm = F.normalize(target_signatures,p=2,dim=1)
        cosine_scores = torch.matmul(Z_train_norm,S_norm[label_idx,:])
#        cosine_angles = torch.acos(cosine_scores)*(180/3.14)
        
        ## Plot histogram of angles
        target_idx = np.where(Y_train.detach().cpu()==label_idx.detach().cpu())[0]
        bg_idx = np.where(Y_train.detach().cpu()!=label_idx.detach().cpu())[0]
        cosine_angles = cosine_scores.detach().cpu().numpy() + 1
        cosine_angles_target = cosine_angles[target_idx]
        cosine_angles_bg = cosine_angles[bg_idx]
        
        min_target_sim = np.min(cosine_angles_target)
        max_bg_sim = np.max(cosine_angles_bg)
        min_target_sim = np.round(min_target_sim,decimals=3)
        max_bg_sim = np.round(max_bg_sim,decimals=3)
        
        ## Make histograms
        plt.figure()
        legend0 = 'Target Class ' + str(label_idx.detach().cpu().numpy())
        legend1 = 'Background Class ' + str(label_idx.detach().cpu().numpy())
        plt.hist(cosine_angles_bg,bins=np.arange(-0.01,2.01,0.02), density=True, color='blue', label=legend1, stacked=True, alpha=0.7, edgecolor='black', linewidth=1.2)
        plt.hist(cosine_angles_target,bins=np.arange(-0.01,2.01,0.02), density=True, color='darkorange', label=legend0, stacked=True, alpha=0.6, edgecolor='black', linewidth=1.2)
        title = 'Cosine Similarity to Class ' + str(label_idx.detach().cpu().numpy()) +' Representative'
        plt.gca().set(title=title, xlabel='Cosine Similarity + 1')
        plt.legend(loc='upper left')
        plt.figtext(0.15,0.7,'min target sim: %.3f' % min_target_sim)
        plt.figtext(0.15,0.66,'max bg sim: %.3f' % max_bg_sim)
        savename = sub_cos_path + '/class_' + str(label_idx.detach().cpu().numpy()) + '.png'
        plt.savefig(savename)
        # plt.show()
        
        ########################### ACE Similarity ############################
        
        ## Whiten data
        Z_train_centered = Z_train - background_means
        U, D, _ = torch.svd(torch.mm(background_covs,background_covs.T))
        D_inv_half = torch.diag(torch.pow(D,-0.5))
        E = torch.matmul(D_inv_half,U.T)
        
        Z_hat_train = torch.matmul(Z_train_centered,E)
        S_hat = torch.matmul(target_signatures, E)
        
        ## Compute ACE Scores
        Z_hat_hat_train = F.normalize(Z_hat_train,p=2,dim=1)
        S_hat_hat = F.normalize(S_hat,p=2,dim=1)
        ace_scores = torch.matmul(Z_hat_hat_train,S_hat_hat[label_idx.detach().cpu().numpy(),:])
        ace_scores = ace_scores + 1
        
        ## ACE Score histograms
        target_idx = np.where(Y_train.detach().cpu()==label_idx.detach().cpu())[0]
        bg_idx = np.where(Y_train.detach().cpu()!=label_idx.detach().cpu())[0]
        cosine_angles = ace_scores.detach().cpu().numpy()
        cosine_angles_target = cosine_angles[target_idx]
        cosine_angles_bg = cosine_angles[bg_idx]
        
        min_target_sim = np.min(cosine_angles_target)
        max_bg_sim = np.max(cosine_angles_bg)
        min_target_sim = np.round(min_target_sim,decimals=3)
        max_bg_sim = np.round(max_bg_sim,decimals=3)
        
        ## Make histograms
        plt.figure()
        legend0 = 'Target Class ' + str(label_idx.detach().cpu().numpy())
        legend1 = 'Background Class ' + str(label_idx.detach().cpu().numpy())
        plt.hist(cosine_angles_bg,bins=np.arange(-0.01,2.01,0.02), density=True, color='blue', label=legend1, stacked=True, alpha=0.7, edgecolor='black', linewidth=1.2)
        plt.hist(cosine_angles_target,bins=np.arange(-0.01,2.01,0.02), density=True, color='darkorange', label=legend0, stacked=True, alpha=0.6, edgecolor='black', linewidth=1.2)
        title = 'ACE Similarity to Class ' + str(label_idx.detach().cpu().numpy()) +' Representative'
        plt.gca().set(title=title, xlabel='ACE Similarity + 1')
        plt.legend(loc='upper left')
        plt.figtext(0.15,0.7,'min target ACE: %.3f' % min_target_sim)
        plt.figtext(0.15,0.66,'max bg ACE: %.3f' % max_bg_sim)
        savename = sub_ace_path + '/class_' + str(label_idx.detach().cpu().numpy()) + '.png'
        plt.savefig(savename)
        # plt.show()
    
    return

def main(Params):
    """
    ******************************************************************
        *  Func:    main()
        *  Desc:    Generates histogram visualizations of the cosine 
        *           similarities between samples and their corresponding
        *           class representatives in the embedding space.
        *
        *  Inputs:    
        *           Params:
        *               Dictionary of global parameters. 
        *     
    ******************************************************************
    """
    for Dataset in Params['data_selection']:
        for regularization_method in Params['regularization_method']:    
            for weight in Params['current_weight']:
                for run in Params['current_run']:
                    for split in Params['train_test_split']:
            
                        ##################### Define embedding/classification model #####################
                        ## Locate model
                        path = Params['base_path'] + Dataset + '/GAP_resnet18/' + regularization_method + '/Weight_' + str(weight) + '/Embed_512D/Run_' + str(run) 
                        params['param_dict'] = path + '/train_dict.pkl'
                        params['model_params'] = path + '/Best_Weights.pt'
                    
                        ## Extra parameters
                        model_name = Params['Model_names'][Dataset]  
                        num_classes = Params['num_classes'][Dataset]
                        numBins = Params['numBins']
                        num_feature_maps = Params['out_channels'][model_name]
                        feat_map_size = Params['feat_map_size']
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")              
                        current_dim = Params['embed_dim'][0]
                        
                        ## Define Model
                        histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                                                 Params['kernel_size'][model_name],
                                                                 num_bins=numBins, stride=Params['stride'],
                                                                 normalize_count=Params['normalize_count'],
                                                                 normalize_bins=Params['normalize_bins'])
                    
                        # Initialize the histogram model for this run
                        model_ft, input_size = initialize_model(model_name, 
                                                                 num_classes,Params['in_channels'][model_name], 
                                                                 num_feature_maps,
                                                                 feature_extract=Params['feature_extraction'],
                                                                 histogram=Params['histogram'],
                                                                 histogram_layer=histogram_layer,
                                                                 parallel=Params['parallel'],
                                                                 use_pretrained=Params['use_pretrained'],
                                                                 add_bn=Params['add_bn'],
                                                                 scale=Params['scale'],
                                                                 feat_map_size=feat_map_size,
                                                                 embed_dim=current_dim,
                                                                 loss=regularization_method,
                                                                 weight=weight)
                    
                        try:
                            ## Load pretrained model parameters
                            model_ft = nn.DataParallel(model_ft)
                            model_ft.load_state_dict(torch.load(params['model_params']))
                            model_ft.eval()
                    
                        except OSError:
                            continue
                            
                        # Send the model to GPU if available, use multiple if available
                        if torch.cuda.device_count() > 1:
                            print("Using", torch.cuda.device_count(), "GPUs!")
                            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                            model_ft = nn.DataParallel(model_ft)
                        model_ft = model_ft.to(device)
                    
                        # Create training and validation dataloaders
                        print("Initializing Datasets and Dataloaders...")
                        Params['batch_size'] = {'train': 128, 'val': 128, 'test': 128}
                        dataloaders_dict = Prepare_DataLoaders(Params, 1, input_size=input_size)
                        
                        
                        ############  Visualize distribution of angles to target reps #############
                        with open(params['param_dict'], 'rb') as f:
                            data_dict = pickle.load(f, encoding='bytes')
                        
                        best_epoch = data_dict['best_epoch']
                        
                        
                        if ((regularization_method == 'LACE') or (regularization_method == 'LACES') or (regularization_method == 'LACER')):
                        
                            ## Get ACE statistics
                            background_means = data_dict['background_means'][best_epoch]
                            background_covs = data_dict['background_covs'][best_epoch]
                            target_signatures = data_dict['target_signatures'][best_epoch]
                            
                            ## Create directories to save figures
                            angle_dist_path = path + '/angle_distributions'
                            try:
                                os.mkdir(angle_dist_path)
                            except OSError:
                                print('')
                                    
                            cos_path = angle_dist_path + '/cosine_sim'
                            try:
                                os.mkdir(cos_path)
                            except OSError:
#                                shutil.rmtree(cos_path, ignore_errors=True)
#                                os.mkdir(cos_path)
                                print('')
                            
                            ace_path = angle_dist_path + '/ace_sim'
                            try:
                                os.mkdir(ace_path)
                            except OSError:
#                                shutil.rmtree(ace_path, ignore_errors=True)
#                                os.mkdir(ace_path)
                                print('')
                                
                            whiten_path = angle_dist_path + '/whiten_data'
                            try:
                                os.mkdir(whiten_path)
                            except OSError:
                                shutil.rmtree(whiten_path, ignore_errors=True)
                                os.mkdir(whiten_path)
                                
                            non_whiten_path = angle_dist_path + '/non_whiten_data'
                            try:
                                os.mkdir(non_whiten_path)
                            except OSError:
                                shutil.rmtree(non_whiten_path, ignore_errors=True)
                                os.mkdir(non_whiten_path)
                            
                            ## Generate visualizations
                            # visulaize_angle_dist_ace(background_means, background_covs, target_signatures, model_ft, dataloaders_dict, split, regularization_method, cos_path, ace_path)
                            visulaize_embeddings(background_means, background_covs, target_signatures, 
                             model_ft, dataloaders_dict,regularization_method, whiten_path, non_whiten_path, split)
    
                        else:
                            
                            ## Create directories to save figures
                            angle_dist_path = path + '/angle_distributions'
                            try:
                                os.mkdir(angle_dist_path)
                            except OSError:
                                print('')
                                    
                            cos_path = angle_dist_path + '/cosine_sim'
                            try:
                                os.mkdir(cos_path)
                            except OSError:
#                                shutil.rmtree(cos_path, ignore_errors=True)
#                                os.mkdir(cos_path)
                                print('')
                            ## Generate visualizations
                            visulaize_angle_dist_non_ace(model_ft, dataloaders_dict, split, regularization_method, cos_path)

                    
###############################################################################
############################## Main ###########################################
###############################################################################

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    
    params['base_path'] = './Results/'
#    params['regularization_method'] = ['AMC', 'arcface', 'center', 'cosface' ,'LACES', 'softmax', 'sphereface']
    params['regularization_method'] = ['LACES']
    params['current_weight'] = [0] #[0,0.25,0.5,0.75,1]
    params['current_run'] = [1]
    params['train_test_split'] = ['val']
    params['data_selection'] = ['FashionMNIST']
#    params['data_selection'] = ['FashionMNIST', 'CIFAR10', 'SVHN']

    main(params)
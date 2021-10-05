# -*- coding: utf-8 -*-
"""
Save results from training/testing model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
from sklearn import manifold
from sklearn.preprocessing import normalize


## PyTorch dependencies
import torch
import torch.nn.functional as F


def save_results(train_dict, test_dict, split, Network_parameters, num_params,
                 regulation_method,dataloaders,model_ft,feat_dim=2,weight=None,
                 device='cuda'):
    
    # Baseline model
    filename = '{}/{}/{}/{}/{}/Weight_{}/Embed_{}D/Run_{}/'.format(Network_parameters['folder'],
                                             Network_parameters['mode'],
                                             Network_parameters['Dataset'],
                                             Network_parameters['Model_name'],
                                             regulation_method,
                                             weight,feat_dim,split+1)

    if not os.path.exists(filename):
        os.makedirs(filename)
        
    #Will need to update code to save everything except model weights to
    # dictionary (use torch save)
    #Save test accuracy
    with open((filename + 'Test_Accuracy.txt'), "w") as output:
        output.write(str(test_dict['test_acc']))
    
    #Save training and testing dictionary, save model using torch
    torch.save(train_dict['best_model_wts'], filename + 'Best_Weights.pt')
    
    #Generate 3D visualization
    # if (feat_dim==2) or (feat_dim == 3):
    transforms = dataloaders['test'].dataset.transform
    
    #Break up visual for train/validation (indexing issue)
    dataloaders['train_full'].dataset.transforms = transforms
    embeds, labels = get_embeds(model_ft, train_dict, 
                                    dataloaders['train_full'],device=device)
    for phase in ['train','val']:
        plot(embeds, labels, dataloaders['TSNE'][phase],
             fig_path=filename + '{}_{}_embedding.png'.format(phase,regulation_method))
        
    #Test data visualization
    embeds, labels = get_embeds(model_ft, train_dict, dataloaders['test'],device=device)
    plot(embeds, labels, dataloaders['TSNE']['test'],
     fig_path=filename + '{}_{}_embedding.png'.format('test',regulation_method))
    
    #Remove model from training dictionary
    train_dict.pop('best_model_wts')
    output_train = open(filename + 'train_dict.pkl','wb')
    pickle.dump(train_dict,output_train)
    output_train.close()
    
    output_test = open(filename + 'test_dict.pkl','wb')
    pickle.dump(test_dict,output_test)
    output_test.close()

        
        
def get_embeds(model, train_dict, loader,device='cuda'):
    
    model.load_state_dict(train_dict['best_model_wts'])
    model = model.to(device).eval()
    full_embeds = []
    full_labels = []
    with torch.no_grad():
        for i, (inputs, labels, index) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            full_labels.append(labels.cpu().detach().numpy())
            _,embeds,_ = model(inputs,labels)
            full_embeds.append(embeds.detach().cpu().numpy())
    model = model.cpu()
    return np.concatenate(full_embeds), np.concatenate(full_labels)
   
def plot(embeds, labels, indices, fig_path='./example.pdf'):

    fig = plt.figure(figsize=(10,10))
    
    #Select images to visualize
    embeds = embeds[indices]
    labels = labels[indices]
    
    #Perform tSNE if embed dim greater than 2 or 3 dimension
    embed_dim = embeds.shape[1]
    if embed_dim > 3:
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=0,
                             metric = 'cosine', square_distances=True)
        embeds = tsne.fit_transform(embeds)
        embed_dim = 3
    
    #L2 normalize 
    embeds = normalize(embeds)
    
    if embed_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    
        #L2 normalize
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
        ax.scatter(embeds[:,0], embeds[:,1], embeds[:,2], c=labels, s=20)
    
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        # ax.grid(False)
    
    else:
        ax = fig.add_subplot(111)
    
        # Create a circle
        t = np.linspace(0,np.pi*2,100)
        ax.plot(np.cos(t), np.sin(t), color='w', alpha=.3, linewidth=0)
        ax.scatter(embeds[:,0], embeds[:,1], c=labels, s=20)
    
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    

 
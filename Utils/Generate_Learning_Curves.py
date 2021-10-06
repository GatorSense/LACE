# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:45:03 2020
Function to generate loss and accuracy curves
@author: jpeeples
"""

import matplotlib.pyplot as plt
import pdb

def Plot_Learning_Curves(train_acc,train_loss,val_acc,val_loss,best_epoch,
                         sub_dir,weight):
    
    # Turn interactive plotting off, don't show plots
    plt.ioff()
    
    #For each type of loss in dictionary, plot training and validation loss
    # in each subplots
    
    loss_fig = plt.figure(figsize=(14,6))
    
    count = 0
    
    #Need to account for not having weight when saving embedding and classification loss
    # Will remove when network functions is updated
    # May not need because I want to see the loss as the model changes
    weight_dict = {'total': None, 'class_loss': 1-weight, 'embed_loss': weight}
    for key in train_loss.keys():
        
        loss_ax = loss_fig.add_subplot(1, 3, count + 1)
        loss_ax.plot(train_loss[key])
        loss_ax.plot(val_loss[key])
        loss_ax.plot([best_epoch],val_loss[key][best_epoch], 
                            marker='o', markersize=3, color='red')
        if weight_dict[key] is not None:
            loss_ax.set_title(key.capitalize() + ' (' + str(weight_dict[key]) + ')')
        else:
            loss_ax.set_title(key.capitalize())

        loss_ax.set_xlabel('Epochs')
        loss_ax.set_ylabel('Error')
        # loss_ax[count].legend(['Training', 'Validation', 'Best Epoch'], loc='upper right')
        count += 1
    
    loss_fig.tight_layout(pad=2.0,rect=[0, 0.03, 1, 0.95])
    loss_fig.subplots_adjust(right=0.87) 
    loss_ax.legend(['Training', 'Validation', 'Best Epoch'],
              bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.suptitle('Learning Curves for {} Epochs For Weight = {}'.format(len(train_loss[key]),weight))
    loss_fig.savefig((sub_dir + 'Loss Curves.png'), dpi=loss_fig.dpi)
    plt.close(loss_fig)
        
    # visualize results
    acc_fig = plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.plot([best_epoch], val_acc[best_epoch], 
             marker='o', markersize=3, color='red')
    plt.suptitle('Accuracy for {} Epochs'.format(len(train_acc)))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation', 'Best Epoch'], loc='best')
    acc_fig.savefig((sub_dir + 'Accuracy Curve.png'), dpi=acc_fig.dpi)
    plt.close(acc_fig)
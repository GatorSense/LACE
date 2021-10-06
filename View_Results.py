# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
import pickle
import scipy.stats as scp
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Generate_TSNE_visual import Generate_TSNE_visual
from Texture_information import Class_names
from Demo_Parameters import Network_parameters as Results_parameters
from Utils.Network_functions import initialize_model
from Prepare_Data import Prepare_DataLoaders
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Confusion_mats import plot_confusion_matrix, plot_avg_confusion_matrix
from Utils.Generate_Learning_Curves import Plot_Learning_Curves

# Location of experimental results
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fig_size = Results_parameters['fig_size']
font_size = Results_parameters['font_size']

# Set up number of runs and class/plots names
NumRuns = Results_parameters['Splits'][Results_parameters['Dataset']]
plot_name = Results_parameters['Dataset'] + ' Test Confusion Matrix'
avg_plot_name = Results_parameters['Dataset'] + ' Test Average Confusion Matrix'
class_names = Class_names[Results_parameters['Dataset']]

# Name of dataset
Dataset = Results_parameters['Dataset']

# Model(s) to be used
model_name = Results_parameters['Model_names'][Dataset]

# Number of classes in dataset
num_classes = Results_parameters['num_classes'][Dataset]

# Number of runs and/or splits for dataset
numRuns = Results_parameters['Splits'][Dataset]

# Number of bins and input convolution feature maps after channel-wise pooling
numBins = Results_parameters['numBins']
num_feature_maps = Results_parameters['out_channels'][model_name]

# Local area of feature map after histogram layer
feat_map_size = Results_parameters['feat_map_size']
# Parse through files and plot results
count = 0

for Dataset in Results_parameters['data_selection']:
    for regularization_method in Results_parameters['regularization_method']:    
        for weight in Results_parameters['current_weight']:
                count += 1

                # Initialize arrays for results
                cm_stack = np.zeros((len(class_names), len(class_names)))
                cm_stats = np.zeros((len(class_names), len(class_names), NumRuns))
                FDR_scores = np.zeros((len(class_names), NumRuns))
                log_FDR_scores = np.zeros((len(class_names), NumRuns))
                accuracy = np.zeros(NumRuns)
                MCC = np.zeros(NumRuns)
                print((('Generating results for {} of {} ' +
                        'Weights').format(count, len(Results_parameters['weights']))))

                # Set directory location for experiments
                if (Results_parameters['histogram']):
                    if (Results_parameters['parallel']):
                        weight_dir = (Results_parameters['folder'] + '/' + Results_parameters['mode']
                                      + '/' + Results_parameters['Dataset'] + '/'
                                      + Results_parameters[
                                          'hist_model'] + '/' + divergence_method + '/' + 'alpha_' + str(
                                    alpha) + '/Weight_' + str(weight)
                                      + '/Embed_' + str(dimension) + 'D'
                                      + '/Parallel/')

                    else:
                        weight_dir = (Results_parameters['folder'] + '/' + Results_parameters['mode']
                                      + '/' + Results_parameters['Dataset'] + '/'
                                      + Results_parameters[
                                          'hist_model'] + '/' + divergence_method + '/' + 'alpha_' + str(
                                    alpha) + '/Weight_' +
                                      str(weight) + '/Embed_' + str(dimension)
                                      + 'D' '/Inline/')
                # Baseline model
                else:
                    weight_dir = (Results_parameters['folder'] + '/' + Results_parameters['mode']
                                  + '/' + Results_parameters['Dataset'] + '/GAP_' +
                                  Results_parameters['Model_names'][Results_parameters['Dataset']]
                                  + '/' + divergence_method + '/' + 'alpha_' + str(alpha) + '/Weight_' + str(
                                weight) + '/Embed_' +
                                  str(dimension) + 'D/')
                silhouette_scores_train = []
                silhouette_scores_test = []
                silhouette_scores_c_train = []
                silhouette_scores_c_test = []
                CH_scores_train = []
                CH_scores_test = []
                for split in range(0, NumRuns):

                    sub_dir = weight_dir + 'Run_' + str(split + 1) + '/'
                    # Load training and testing files (Python)
                    train_pkl_file = open(sub_dir + 'train_dict.pkl', 'rb')
                    train_dict = pickle.load(train_pkl_file)
                    train_pkl_file.close()

                    test_pkl_file = open(sub_dir + 'test_dict.pkl', 'rb')
                    test_dict = pickle.load(test_pkl_file)
                    test_pkl_file.close()

                    # Remove pickle files
                    del train_pkl_file, test_pkl_file

                    # #Load model
                    histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                                     Results_parameters['kernel_size'][model_name],
                                                     num_bins=numBins, stride=Results_parameters['stride'],
                                                     normalize_count=Results_parameters['normalize_count'],
                                                     normalize_bins=Results_parameters['normalize_bins'])

                    # Initialize the histogram model for this run
                    model, input_size = initialize_model(model_name, num_classes,
                                                         Results_parameters['in_channels'][model_name],
                                                         num_feature_maps,
                                                         feature_extract=Results_parameters['feature_extraction'],
                                                         histogram=Results_parameters['histogram'],
                                                         histogram_layer=histogram_layer,
                                                         parallel=Results_parameters['parallel'],
                                                         use_pretrained=Results_parameters['use_pretrained'],
                                                         add_bn=Results_parameters['add_bn'],
                                                         scale=Results_parameters['scale'],
                                                         feat_map_size=feat_map_size,
                                                         embed_dim=dimension)

                    # Set device to cpu or gpu (if available)
                    device_loc = torch.device(device)

                    # Generate learning curves
                    Plot_Learning_Curves(train_dict['train_acc_track'],
                                         train_dict['train_error_track'],
                                         train_dict['val_acc_track'],
                                         train_dict['val_error_track'],
                                         train_dict['best_epoch'],
                                         sub_dir, weight)

                    # If parallelized, need to set change model
                    if Results_parameters['Parallelize']:
                        model = nn.DataParallel(model)

                    # model.load_state_dict(train_dict['best_model_wts'])
                    model.load_state_dict(torch.load(sub_dir + 'Best_Weights.pt'
                                                     ,map_location=device_loc))
                    model = model.to(device)

                    dataloaders_dict = Prepare_DataLoaders(Results_parameters, split,
                                                           input_size=input_size)

                    if (Results_parameters['TSNE_visual']):
                        print("Initializing Datasets and Dataloaders...")

                        dataloaders_dict = Prepare_DataLoaders(Results_parameters, split,
                                                               input_size=input_size)
                        print('Creating TSNE Visual...')

                        # Generate TSNE visual
                        FDR_scores[:, split], log_FDR_scores[:, split] = Generate_TSNE_visual(
                            dataloaders_dict,
                            model.features, sub_dir, device, class_names,
                            histogram=Results_parameters['histogram'],
                            Separate_TSNE=Results_parameters['Separate_TSNE'])
                    # Create CM for testing data
                    cm = confusion_matrix(test_dict['GT'], test_dict['Predictions'])
                    # Create classification report
                    report = classification_report(test_dict['GT'], test_dict['Predictions'],
                                                   target_names=class_names, output_dict=True)
                    # Convert to dataframe and save as .CSV file
                    df = pd.DataFrame(report).transpose()
                    # Save to CSV
                    df.to_csv((sub_dir + 'Classification_Report.csv'))

                    # Confusion Matrix
                    np.set_printoptions(precision=2)
                    fig4, ax4 = plt.subplots(figsize=(fig_size, fig_size))
                    plot_confusion_matrix(cm, classes=class_names, title=plot_name, ax=ax4,
                                          fontsize=font_size)
                    fig4.savefig((sub_dir + 'Confusion Matrix.png'), dpi=fig4.dpi)
                    plt.close(fig4)
                    cm_stack = cm + cm_stack
                    cm_stats[:, :, split] = cm

                    # Get accuracy of each cm
                    accuracy[split] = 100 * sum(np.diagonal(cm)) / sum(sum(cm))
                    # Write to text file
                    with open((sub_dir + 'Accuracy.txt'), "w") as output:
                        output.write(str(accuracy[split]))

                    # Compute Matthews correlation coefficient
                    MCC[split] = matthews_corrcoef(test_dict['GT'], test_dict['Predictions'])

                    # Write to text file
                    with open((sub_dir + 'MCC.txt'), "w") as output:
                        output.write(str(MCC[split]))
                    directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'
                    # Visualize embeddings as model changes
                    if (dimension == 2 or
                            dimension == 3):
                        x = []
                        for batch in train_dict['embedding_epochs']:
                            x.extend(batch['train'])
                        y = []
                        for batch in train_dict['embedding_labels']:
                            y.extend(batch['train'])

                    print('**********Run ' + str(split + 1) + ' Finished**********')

                directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'
                np.set_printoptions(precision=2)
                fig5, ax5 = plt.subplots(figsize=(fig_size, fig_size))
                plot_avg_confusion_matrix(cm_stats, classes=class_names,
                                          title=avg_plot_name, ax=ax5, fontsize=font_size)
                fig5.savefig((directory + 'Average Confusion Matrix.png'), dpi=fig5.dpi)
                plt.close()

                # Write to text file
                with open((directory + 'Overall_Cluster_Metrics.txt'), "w") as output:
                    output.write('Average training Silhouette score(Euclidean): ' + str(
                        np.mean(silhouette_scores_train)) + ' Std: ' + str(np.std(silhouette_scores_train)) + '\n')
                    output.write('Average testing Silhouette score(Euclidean): ' + str(
                        np.mean(silhouette_scores_test)) + ' Std: ' + str(np.std(silhouette_scores_test)) + '\n')
                    output.write('Average training Silhouette score(Cosine): ' + str(
                        np.mean(silhouette_scores_c_train)) + ' Std: ' + str(np.std(silhouette_scores_c_train)) + '\n')
                    output.write('Average testing Silhouette score(Cosine): ' + str(
                        np.mean(silhouette_scores_c_test)) + ' Std: ' + str(np.std(silhouette_scores_c_test)) + '\n')
                    output.write('Average training CH score: ' + str(np.mean(CH_scores_train)) + ' Std: ' + str(
                        np.std(CH_scores_train)) + '\n')
                    output.write('Average testing CH score: ' + str(np.mean(CH_scores_test)) + ' Std: ' + str(
                        np.std(CH_scores_test)) + '\n')
                # Write to text file
                with open((directory + 'Overall_Accuracy.txt'), "w") as output:
                    output.write('Average accuracy: ' + str(np.mean(accuracy)) + ' Std: ' + str(np.std(accuracy)))

                # Write to text file
                with open((directory + 'Overall_MCC.txt'), "w") as output:
                    output.write('Average MCC: ' + str(np.mean(MCC)) + ' Std: ' + str(np.std(MCC)))

                # Write to text file
                with open((directory + 'training_Overall_FDR.txt'), "w") as output:
                    output.write('Average FDR: ' + str(np.mean(FDR_scores, axis=1))
                                 + ' Std: ' + str(np.std(FDR_scores, axis=1)))
                with open((directory + 'training_Overall_Log_FDR.txt'), "w") as output:
                    output.write('Average FDR: ' + str(np.mean(log_FDR_scores, axis=1))
                                 + ' Std: ' + str(np.std(log_FDR_scores, axis=1)))

                # Write list of accuracies and MCC for analysis
                np.savetxt((directory + 'List_Accuracy.txt'), accuracy.reshape(-1, 1), fmt='%.2f')
                np.savetxt((directory + 'List_MCC.txt'), MCC.reshape(-1, 1), fmt='%.2f')

                # save weight 0 accuracies for t-tests
                if (weight == 0):
                    weight_0_accuracies = accuracy

                # Write t-test Results
                with open((directory + 'T-Test.txt'), "w") as output:
                    output.write(
                        'T-Test for accuracies of this weight compared with those from neural net with no embedding loss term (weight 0) \n')
                    output.write(str(scp.ttest_ind(weight_0_accuracies, accuracy)))
                np.savetxt((directory + 'training_List_FDR_scores.txt'), FDR_scores, fmt='%.2E')
                np.savetxt((directory + 'training_List_log_FDR_scores.txt'), log_FDR_scores, fmt='%.2f')
                plt.close("all")
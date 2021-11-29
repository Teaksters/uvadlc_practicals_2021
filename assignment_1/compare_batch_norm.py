################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import torch
import torch.nn as nn
import torch.optim as optim

import pickle
import matplotlib.pyplot as plt
import time
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested
    epochs, lr, batch_size, seed, data_dir = 20, 0.1, 128, 42, 'data/'
    hidden_dims = [[128], [256, 128], [512, 256, 128]]
    use_batch_norms = [True, False]

    results = dict()
    for h_dim in hidden_dims:
        for batch_norm in use_batch_norms:
            print('batch_norm: ', batch_norm)
            start_time = time.time()
            m, v_acc, test_acc, logging_dict = train_mlp_pytorch.train(h_dim,
                                                            lr, batch_norm,
                                                            batch_size, epochs,
                                                            seed, data_dir)
            print('that took: ', str(time.time() - start_time), 'seconds')
            results[(tuple(h_dim), batch_norm)] = logging_dict

    for key in results:
        print(key)
        print(results[key])

    pickle.dump( results, open( results_filename, "wb" ) )
    # Feel free to add any additional functions, such as plotting of the loss curve here
    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    colors = ['#fc8d59','#762a83','#91bfdb']
    results = pickle.load( open( results_filename, "rb" ) )
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("validation accuracy", loc='center')
    axs[1].set_title("training accuracy", loc='center')
    for i, key in enumerate(results):
        h_layers, b_norm = key[0], key[1]
        # Prepare data
        v = results[key]['val_acc']
        t = results[key]['train_acc']
        x = np.arange(len(v))

        if b_norm:
            if i < 2:
                axs[0].plot(x, v, color=colors[int(i/2)], ls='--', label='Batch Norm ')
                axs[1].plot(x, t, color=colors[int(i/2)], ls='--')
            else:
                axs[0].plot(x, v, color=colors[int(i/2)], ls='--')
                axs[1].plot(x, t, color=colors[int(i/2)], ls='--')
        else:
            if i < 2:
                axs[0].plot(x, v, color=colors[int(i/2)], label='Regular')
                axs[1].plot(x, t, color=colors[int(i/2)], label=str(h_layers))
            else:
                axs[0].plot(x, v, color=colors[int(i/2)])
                axs[1].plot(x, t, color=colors[int(i/2)], label=str(h_layers))
    #
    #
    # plt.title('Train and validation Accuracy')
    # axs[1].set_xlabel('epoch')
    #
    # ax1.plot(np.arange(len(logging_dict['loss'])), logging_dict['loss'],
    #          label='batch loss')
    # ax1.legend()
    #
    # ax2.plot(np.arange(len(logging_dict['val_acc'])), logging_dict['val_acc'],
    #          label='validation accuracy (epoch)')
    # ax2.legend()

    axs[0].set_ylabel('Accuracy')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xlabel('epoch')
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'test.p'
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)

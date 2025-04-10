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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.mean(predictions == targets)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy

def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    targets = np.array([])
    predictions = np.array([])
    for x, y in data_loader:
        x = np.reshape(x, [x.shape[0], -1])
        pred = model.forward(x)
        targets = np.concatenate((targets, y), axis=0)
        predictions = np.vstack([predictions, pred]) if predictions.size else pred
    avg_accuracy = accuracy(predictions, targets)

    # Make sure to remove accumulated gradients
    model.clear_cache()
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Initialize model and loss module
    features = 32 * 32 * 3
    classes = 10
    model = MLP(features, hidden_dims, classes)
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    logging_dict = {'val_acc': [],
                    'loss': []}
    best_accuracy = 0
    for epoch in range(epochs):
        print('epoch: ', str(epoch + 1), '/', str(epochs))
        # Training loop
        for x, y in cifar10_loader['train']:
            # Forward pass through the model
            x = np.reshape(x, [x.shape[0], -1])
            pred = model.forward(x)
            loss = loss_module.forward(pred, y)
            logging_dict['loss'].append(loss)

            # Backward pass through the model
            d_loss = loss_module.backward(pred, y)
            model.backward(d_loss)

            # Update parameters of linear layers with gradients
            for i in range(0, len(model.layers), 2):
                model.layers[i].params['weight'] -= lr * model.layers[i].grads['weight']
                model.layers[i].params['bias'] -= lr * model.layers[i].grads['bias']

            # Clear gradients for next pass.
            model.clear_cache()

        # Per epoch validation evaluation
        avg_accuracy = evaluate_model(model, cifar10_loader['validation'])
        logging_dict['val_acc'].append(avg_accuracy)

        # Store best model
        if avg_accuracy > best_accuracy:
            best_model = deepcopy(model)
            best_accuracy = avg_accuracy

    # Test best model
    logging_dict['test_acc'] = evaluate_model(best_model, cifar10_loader['test'])
    test_accuracy = logging_dict['test_acc']
    val_accuracies = logging_dict['val_acc']
    model = best_model
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    m, v_acc, test_acc, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    fig, ((ax1), (ax2)) = plt.subplots(2, 1)
    plt.title('Train Loss and Validation Accuracy Numpy')

    ax1.plot(np.arange(len(logging_dict['loss'])), logging_dict['loss'],
             label='batch loss')
    ax1.legend()

    ax2.plot(np.arange(len(logging_dict['val_acc'])), logging_dict['val_acc'],
             label='validation accuracy (epoch)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

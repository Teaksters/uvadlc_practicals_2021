###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set

from tqdm import tqdm
import pickle


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name.

    Args:
        model_name: Name of the model architecture to be returned.
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18',
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*3, num_classes)
            )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
            cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train, val = get_train_validation_set(data_dir)

    # Initialize dataloaders
    tr_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                drop_last=False)
    val_loader = data.DataLoader(val, batch_size=batch_size, shuffle=False,
                                 drop_last=False)

    # Initialize the optimizers and learning rate scheduler.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[90, 135],
                                                     gamma=0.1)

    # Initialize loss module
    loss_module = nn.CrossEntropyLoss()

    # Save the best model, and remember to use the lr scheduler.
    val_scores = []
    train_losses, train_scores = [], []
    best_val_epoch = -1
    for epoch in range(epochs):
        # Training the model with the train set
        model.train()
        true_preds, count = 0., 0
        t = tqdm(tr_loader, leave=False)
        for imgs, labels in t:
            # Reset gradients and push data to model location
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward input through model and calculate loss
            preds = model(imgs)
            loss = loss_module(preds, labels)

            # Perform backpropogation with gradients of the loss
            loss.backward()
            optimizer.step()

            # Record training statistics
            true_preds += (preds.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]
            t.set_description(f"Epoch {epoch+1}: loss={loss.item():4.2f}")
            train_losses.append(loss.item())
        train_acc = true_preds / count
        train_scores.append(train_acc)

        # Update learning rate
        scheduler.step()

        # Evaluate model on validation set
        val_acc = evaluate_model(model, val_loader, device)
        val_scores.append(val_acc)
        print(f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {val_acc*100.0:05.2f}%")

        # Store the best performing model
        print(val_scores, best_val_epoch)
        if len(val_scores) == 1 or val_acc > val_scores[best_val_epoch]:
            print("\t   (New best performance, saving model...)")
            torch.save(model, checkpoint_name)
            best_val_epoch = epoch


    # Load best model and return it.
    model = torch.load(checkpoint_name)
    model.eval()
    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode
    model.eval()
    true_preds, count = 0., 0

    # Loop over the provided dataset
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        # Evaluate model performance with accuracy measures
        with torch.no_grad():
            preds = model(imgs).argmax(dim=-1)
            true_preds += (preds == labels).sum().item()
            count += labels.shape[0]
    # Normalize the accuracy to be between 0 and 1
    accuracy = true_preds / count

    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test.
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(seed)
    test_results = {}

    # Augmentation settings
    augmentations = [None, gaussian_noise_transform, gaussian_blur_transform,
                     contrast_transform, jpeg_transform]
    severity = np.arange(1, 6)

    # Loop over all types of augmentation and severities
    for augmentation in augmentations:
        print(augmentation)
        for sev in severity:
            # Prepare data loaders
            if augmentation == None:
                test = get_test_set(data_dir)
            else:
                test = get_test_set(data_dir,
                                augmentation=augmentation(severity=sev))
            test_loader = data.DataLoader(test, batch_size=batch_size,
                                          shuffle=False, drop_last=False)

            # Evaluate model performance
            test_acc = evaluate_model(model, test_loader, device)

            # Store results
            if augmentation == None:
                aug = 'None'
            else:
                aug = str(augmentation).split()[1]
            print('Tested: ', aug, sev)
            test_results[(aug, sev)] = test_acc

            # If no augmentation applied no severities is applied
            if augmentation == None: break
    #######################
    # END OF YOUR CODE    #
    #######################
    print('Done!')
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Create a checkpoints folder for model storage
    CHECKPOINT_PATH = 'chkpts'
    RESULT_PATH = 'rslts'
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(RESULT_PATH, exist_ok=True)
    checkpoint_name = os.path.join(CHECKPOINT_PATH, model_name)
    result_name = os.path.join(RESULT_PATH, model_name + '.pkl')

    # Prepare devices and seeds
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(seed)

    # Initialize desired model
    # model = get_model(model_name)
    # model.to(device)

    # # Train model on the CIFAR10 dataset
    # model = train_model(model, lr, batch_size, epochs, data_dir,
    #                     checkpoint_name, device)

    # If using with pretrained checkpoint
    model = torch.load(os.path.join(CHECKPOINT_PATH, model_name))
    model.to(device)
    # Test model with several augmentations and severities
    test_results = test_model(model, batch_size, data_dir, device, seed)

    # Store results for later use
    pickle.dump( test_results, open(result_name, "wb"))
    #######################
    # END OF YOUR CODE    #
    #######################





if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)

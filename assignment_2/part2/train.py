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
# Date Adapted: 2021-11-11
###############################################################################

from datetime import datetime
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel

from torch.utils.tensorboard import SummaryWriter
import os


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


def train(args):
    """
    Trains an LSTM model on a text dataset

    Args:
        args: Namespace object of the command line arguments as
              specified in the main function.

    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here.
    Call the model forward function on the inputs,
    calculate the loss with the targets and back-propagate,
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    args.vocabulary_size = dataset._vocabulary_size

    data_loader = DataLoader(dataset, args.batch_size,
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)

    # Create model
    model = TextGenerationModel(args)
    model = model.to(args.device)
    # Create optimizer
    loss_module = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    counter = 0
    for epoch in range(args.num_epochs):

        # Train loop
        print('epoch: ', epoch, '/', args.num_epochs)
        true_preds, count = 0., 0
        t = tqdm(data_loader, leave=False)
        for i, (x, labels) in enumerate(t):
            x, labels = x.to(args.device), labels.to(args.device)
            optimizer.zero_grad()

            # Make predictions
            preds = model(x)
            preds = model.Softmax(preds)

            # Calculate losses
            loss = torch.zeros(1).to(args.device)
            labels = nn.functional.one_hot(labels,
                    num_classes=args.vocabulary_size).type(torch.FloatTensor).to(args.device)
            for t, preds_t in enumerate(preds):
                loss += loss_module(preds_t, labels[t])
            loss /= preds.shape[0]

            # backpropogation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # Record statistics during training
            counter += 1
            writer.add_scalar("Train Loss", loss.item(), counter)
            true_preds += (preds.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item()
            count += labels.shape[0] * labels.shape[1]
        train_acc = true_preds / count
        print('accuracy: ', train_acc)
        writer.add_scalar("Train Accuracy", train_acc, epoch)

        # Store model during checkpoints
        if epoch + 1 in args.checkpoints:
            model_path = 'chkpts'
            os.makedirs(model_path, exist_ok=True)
            torch.save(model, os.path.join(model_path, args.txt_file[7:-4] + 'ep_' + str(epoch)))



    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')
    parser.add_argument('--checkpoints', type=list, default=[1, 5, 20], help='checkpoints for model storage')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    writer = SummaryWriter(os.path.join('runs', args.txt_file[7:-4]))

    train(args)

    writer.flush()

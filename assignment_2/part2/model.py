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

import math
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """
    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Define parameters
        self.Wgx = nn.Parameter(torch.FloatTensor(self.embed_dim, self.hidden_dim))
        self.Wgh = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.Wix = nn.Parameter(torch.FloatTensor(self.embed_dim, self.hidden_dim))
        self.Wih = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.Wfx = nn.Parameter(torch.FloatTensor(self.embed_dim, self.hidden_dim))
        self.Wfh = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.Wox = nn.Parameter(torch.FloatTensor(self.embed_dim, self.hidden_dim))
        self.Woh = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))

        self.bg = nn.Parameter(torch.zeros((1, self.hidden_dim)))
        self.bi = nn.Parameter(torch.zeros((1, self.hidden_dim)))
        self.bf = nn.Parameter(torch.ones((1, self.hidden_dim)))
        self.bo = nn.Parameter(torch.zeros((1, self.hidden_dim)))
        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        r = 1 / math.sqrt(self.hidden_dim)
        for param in self.parameters():
            if param.shape[0] != 1:
                with torch.no_grad():
                    param.uniform_(-r, r)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Probably need to loop over a dimension of the embeds
        output = torch.FloatTensor(embeds.shape[0], embeds.shape[1], self.hidden_dim)
        self.h = torch.zeros([1, self.hidden_dim])
        self.c = torch.ones([embeds.shape[1], self.hidden_dim])
        for idx, x in enumerate(embeds):
            g = torch.tanh(x @ self.Wgx + self.h @ self.Wgh + self.bg)
            i = torch.sigmoid(x @ self.Wix + self.h @ self.Wih + self.bi)
            f = torch.sigmoid(x @ self.Wfx + self.h @ self.Wfh + self.bf)
            o = torch.sigmoid(x @ self.Wox + self.h @ self.Woh + self.bo)
            self.c = g * i + self.c * f
            # Update stored hidden parameters
            self.h = torch.tanh(self.c) * o
            output[idx] = self.h

        return output
        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """
    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.vocabulary_size = args.vocabulary_size
        self.embedding_size = args.embedding_size
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.LSTM = LSTM(self.lstm_hidden_dim, self.embedding_size)

        # Initialize linear layer
        self.outL = nn.Linear(self.lstm_hidden_dim, self.vocabulary_size)
        self.Softmax = nn.Softmax(dim=2)

        # Embedding Weights
        r = 1 / math.sqrt(self.vocabulary_size)
        self.Emb = nn.Embedding(self.vocabulary_size, self.embedding_size)

        # Push everything to device
        print(args.device)
        self.Emb.to(args.device)
        self.outL.to(args.device)
        self.LSTM.to(args.device)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Embed character in learned embedding
        x = self.Emb(x)

        # Apply LSTM cell
        y_hidden = self.LSTM(x)

        # Use LSTM output to generate characters
        y = self.outL(y_hidden)
        return y
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # Initialize output structure
        with torch.no_grads():
            output = torch.empty([sample_length, batch_size, 1],
                                 dtype=torch.int64)

            # Define a random start
            random_start = torch.randint(0, self.vocabulary_size, [batch_size, 1])
            output[0] = random_start
            pred = None # For variable life-span

            for idx, x in enumerate(output):
                # Generate random output with infinite temperature
                if temperature == float('inf'):
                    pred = torch.randint(0, self.vocabulary_size, [batch_size, 1])

                # Generate model output based on provided temperature
                else:
                    h = self.forward(x)
                    # Temperature of 0 results to determinstic behavior
                    if temperature == 0.:
                        pred = h.argmax(dim=2)

                    # Sample from output with insecurity influenced by temperature
                    else:
                        pred = self.Softmax(h / temperature)
                        pred.squeeze_()
                        pred = torch.multinomial(pred, 1)

                # Update next character input with model prediction
                if idx < sample_length - 1:
                    output[idx + 1] = pred

        return output
        #######################
        # END OF YOUR CODE    #
        #######################

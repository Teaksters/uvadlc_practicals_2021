################################################################################
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
# Date Created: 2020-11-27
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np
import torch.nn.functional as F


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    # Sample noise from normal distribution
    epsilon = torch.normal(torch.zeros(mean.shape),
                           torch.ones(mean.shape)).to(mean.device)

    # Sample latent space with mean, std and noise
    z = mean + epsilon * std
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See Section 1.3 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    KLD = ((torch.exp(log_std).square() + mean.square() - 1 - 2 * log_std) / 2)
    KLD = KLD.sum(dim=-1)
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    # Prepare necessary data types and values
    d = torch.Tensor(list(img_shape[1:])).to(elbo.device)
    log2_e = torch.log2(torch.exp(torch.ones(1))).to(elbo.device)

    # Do the calculation
    bpd = (elbo * log2_e) / torch.prod(d)
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    # Generate needed percentiles
    percentiles = torch.arange(0.5, grid_size, 1) / grid_size

    # Translate to z input type
    Normal_distribution = torch.distributions.normal.Normal(torch.zeros([1]), torch.ones([1]))
    unique_values = Normal_distribution.icdf(percentiles)

    # Create all possible input combinations
    grid_x, grid_y = torch.meshgrid(unique_values, unique_values)
    z = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # Generate sample images
    images = []
    for i in range(128, z.shape[0], 128):
        input = z[i-128:i].to(decoder.device)

        # decode images
        imgs = decoder(input).squeeze()
        images.append(imgs)

    # Also add the final samples that did not fit in batch
    images.append(decoder(z[-(z.shape[0] % 128):].to(decoder.device)).squeeze())

    # Apply softmax and fit to readable image format
    images = torch.cat(images, dim=0).permute(0, 2, 3, 1).flatten(0, 2)
    samples = torch.multinomial(F.softmax(images, dim=1), 1).reshape(z.shape[0], 1, 28, 28)

    # Generate image grid for plotting
    img_grid = make_grid(samples, nrow=grid_size).float() / 15
    print(img_grid.shape)
    return img_grid

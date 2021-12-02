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


def main(args):
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    args.vocabulary_size = dataset._vocabulary_size

    data_loader = DataLoader(dataset, args.batch_size,
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)

    # Create model
    model = TextGenerationModel(args)
    model.to(args.device)
    model.eval()

    # Sample randomly generated text
    output = model.sample(temperature=2.0).squeeze().detach().numpy().T
    output = np.vectorize(dataset._ix_to_char.get)(output)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')

    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    main(args)

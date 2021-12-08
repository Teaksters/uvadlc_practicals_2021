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
import os


TEMPS = [0, 0.5, 1.0, 2.0]
SAMPLE_SIZES = [30, 60, 90]
SAMPLE_BATCHES = 5


def main(args):
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    os.makedirs(args.RES_DIR, exist_ok=True)
    out_f = os.path.join(args.RES_DIR, 'generated.txt')
    with open(out_f, 'w') as f:
        # Test model for setup hyperparameters
        for checkpoint in os.listdir(args.CHKPT_DIR):
            path = os.path.join(args.CHKPT_DIR, checkpoint)
            model = torch.load(path)
            model.to(args.device)
            f.write('############Checkpoint: ' + str(checkpoint) + '############\n')
            for T in TEMPS:
                f.write('############Temp: ' + str(T) + '############\n')
                for size in SAMPLE_SIZES:
                    f.write('############Len: ' + str(size) + '############\n')
                    # Create samples
                    for i in range(SAMPLE_BATCHES):
                        output = model.sample(sample_length=size,
                                        temperature=T).squeeze().detach().cpu().numpy().T
                        output = np.vectorize(dataset._ix_to_char.get)(output)
                        for out in output:
                            sentence = ''.join(out)
                            f.write(sentence + '\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    args = parser.parse_args()
    args.CHKPT_DIR = 'chkpts'
    args.RES_DIR = 'rslts'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)

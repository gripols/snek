import argparse

import chainer
from chainer import backends
import librosa
import numpy as np
from tqdm import tqdm

from lib import spec_utils
from lib import unet

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=2019)
    parser.add_argument('--sr', '-r', type=int, default=44100)
    parser.add_argument('--hop_length', '-H', type=int, default=1024)
    parser.add_argument('--n_fft', '-f', type=int, default=2048)
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.001)
    parser.add_argument('--lr_min', type=float, default=0.0001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.9)
    parser.add_argument('--lr_decay_patience', type=int, default=6)
    parser.add_argument('--batchsize', '-B', type=int, default=4)
    parser.add_argument('--accumulation_steps', '-A', type=int, default=1)
    parser.add_argument('--cropsize', '-C', type=int, default=256)
    parser.add_argument('--patches', '-p', type=int, default=16)
    parser.add_argument('--val_rate', '-v', type=float, default=0.2)
    parser.add_argument('--val_filelist', '-V', type=str, default=None)
    parser.add_argument('--val_batchsize', '-b', type=int, default=4)
    parser.add_argument('--val_cropsize', '-c', type=int, default=256)
    parser.add_argument('--num_workers', '-w', type=int, default=4)
    parser.add_argument('--epoch', '-E', type=int, default=200)
    parser.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    parser.add_argument('--reduction_level', '-L', type=float, default=0.2)
    parser.add_argument('--mixup_rate', '-M', type=float, default=0.0)
    parser.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
    parser.add_argument('--pretrained_model', '-P', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()
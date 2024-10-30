# Merlin 2: Electric Boogaloo
import argparse
import os

import librosa
import torch

from lib import dataset
from lib import netta
from lib import spec_utils

# Set up arg. parser for cmd line args
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1, 
                   help='Specify GPU index (default: -1 for CPU)')
    p.add_argument('--pretrained_model', '-P', type=str, 
                   default='models/baseline.pth', help='Path to the pretrained model file')
    p.add_argument('--mixtures', '-m', required=True, 
                   help='Path to the mixture audio files')
    p.add_argument('--instruments', '-i', required=True, 
                   help='Path to the instrument audio files')
    p.add_argument('--sr', '-r', type=int, default=44100, 
                   help='Sample rate (default: 44100)')
    p.add_argument('--n_fft', '-f', type=int, default=2048, 
                   help='Number of FFT points (default: 2048)')
    p.add_argument('--hop_length', '-H', type=int, default=1024, 
                   help='Number of samples between frames (default: 1024)')
    p.add_argument('--batchsize', '-B', type=int, default=4, 
                   help='Batch size for processing (default: 4)')
    p.add_argument('--cropsize', '-c', type=int, default=256, 
                   help='Crop size for processing (default: 256)')
    p.add_argument('--postprocess', '-p', action='store_true', 
                   help='Flag to apply post-processing on the output')
    
    # Parse the cmd line args
    args = p.parse_args()

    

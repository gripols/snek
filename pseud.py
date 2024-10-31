import argparse
import gc
import os
import random

import chainer
import chainer.functions as F
import numpy as np

from lib import dataset
from lib import spec_utils
from lib import netta

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=2019)
    parser.add_argument('--mixture_dataset', '-m', required=True)
    parser.add_argument('--instrumental_dataset', '-i', required=True)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_min', type=float, default=0.00001)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_decay_interval', type=int, default=5)
    parser.add_argument('--batchsize', '-B', type=int, default=32)
    parser.add_argument('--val_batchsize', '-b', type=int, default=32)
    parser.add_argument('--cropsize', '-c', type=int, default=512)
    parser.add_argument('--epoch', '-E', type=int, default=50)
    parser.add_argument('--inner_epoch', '-e', type=int, default=4)
    parser.add_argument('--mixup', '-M', action='store_true')
    parser.add_argument('--mixup_alpha', '-a', type=float, default=0.4)
    return parser.parse_args()

# Set random seeds for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)
    chainer.global_config.autotune = True

# Prepare file lists for training and validation
def prepare_filelists(mixture_dataset, instrumental_dataset, validation_split=20):
    input_exts = ['.wav', '.m4a', '.3gp', '.oma', '.mp3', '.mp4']
    mixture_files = sorted(
        [fname for fname in os.listdir(mixture_dataset) if os.path.splitext(fname)[1] in input_exts]
    )
    instrumental_files = sorted(
        [fname for fname in os.listdir(instrumental_dataset) if os.path.splitext(fname)[1] in input_exts]
    )
    
    filelist = list(zip(
        [os.path.join(mixture_dataset, fname) for fname in mixture_files],
        [os.path.join(instrumental_dataset, fname) for fname in instrumental_files]
    ))
    
    random.shuffle(filelist)
    return filelist[:-validation_split], filelist[-validation_split:]

# Train model
def train(model, optimizer, X_train, y_train, batchsize, device, mixup, mixup_alpha):
    sum_loss = 0
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batchsize):
        local_perm = perm[i: i + batchsize]
        X_batch = model.xp.asarray(X_train[local_perm])
        y_batch = model.xp.asarray(y_train[local_perm])

        model.cleargrads()
        mask = model(X_batch)
        X_batch = spec_utils.crop_and_concat(mask, X_batch, False)
        y_batch = spec_utils.crop_and_concat(mask, y_batch, False)

        loss = F.mean_absolute_error(X_batch * mask, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(X_batch)

    return sum_loss / len(X_train)

# Validate model
def validate(model, X_valid, y_valid, batchsize, device):
    sum_loss = 0
    perm = np.random.permutation(len(X_valid))
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for i in range(0, len(X_valid), batchsize):
            local_perm = perm[i: i + batchsize]
            X_batch = model.xp.asarray(X_valid[local_perm])
            y_batch = model.xp.asarray(y_valid[local_perm])

            mask = model(X_batch)
            X_batch = spec_utils.crop_and_concat(mask, X_batch, False)
            y_batch = spec_utils.crop_and_concat(mask, y_batch, False)

            inst_loss = F.mean_squared_error(X_batch * mask, y_batch)
            vocal_loss = F.mean_squared_error(X_batch * (1 - mask), X_batch - y_batch)
            loss = inst_loss + vocal_loss
            sum_loss += float(loss.data) * len(X_batch)

    return sum_loss / len(X_valid)

# Main training loop
def main():
    args = parse_args()
    set_random_seed(args.seed)

    # Setup model and optimizer
    model = netta.SpecUNet()
    if args.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup(model)

    # Prepare datasets
    train_filelist, valid_filelist = prepare_filelists(args.mixture_dataset, args.instrumental_dataset)
    X_valid, y_valid = dataset.create_dataset(valid_filelist, args.cropsize, validation=True)

    best_loss = np.inf
    patience_counter = 0

    for epoch in range(args.epoch):
        random.shuffle(train_filelist)
        X_train, y_train = dataset.create_dataset(train_filelist[:100], args.cropsize)
        if args.mixup:
            X_train, y_train = dataset.mixup_generator(X_train, y_train, args.mixup_alpha)
        
        print(f'# Epoch {epoch}')
        train_loss = train(model, optimizer, X_train, y_train, args.batchsize, model.device, args.mixup, args.mixup_alpha)
        valid_loss = validate(model, X_valid, y_valid, args.val_batchsize, model.device)

        print(f'  Training Loss: {train_loss:.6f} | Validation Loss: {valid_loss:.6f}')

        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            model_path = f'models/model_epoch{epoch}.npz'
            chainer.serializers.save_npz(model_path, model)
            print('  * New best model saved')
        else:
            patience_counter += 1

        # Learning rate decay
        if patience_counter >= args.lr_decay_interval:
            patience_counter = 0
            optimizer.alpha = max(optimizer.alpha * args.lr_decay, args.lr_min)
            print(f'  * Learning rate decayed to {optimizer.alpha:.6f}')

        # Clear memory
        del X_train, y_train
        gc.collect()

if __name__ == '__main__':
    main()

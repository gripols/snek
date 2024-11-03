import argparse
from datetime import datetime
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from lib import dataset
from lib import netta
from lib import spec_utils


def setup_logger(name, logfile='LOGFILENAME.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(logfile, encoding='utf8')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def to_wave(spec, n_fft, hop_length, window):
    B, _, N, T = spec.shape
    wave = spec.reshape(-1, N, T)
    wave = torch.istft(wave, n_fft, hop_length, window=window)
    wave = wave.reshape(B, 2, -1)
    return wave


def sdr_loss(y, y_pred, eps=1e-8):
    sdr = (y * y_pred).sum() / (torch.linalg.norm(y) * torch.linalg.norm(y_pred) + eps)
    return -sdr


def weighted_sdr_loss(y, y_pred, n, n_pred, eps=1e-8):
    y_sdr = (y * y_pred).sum() / (torch.linalg.norm(y) * torch.linalg.norm(y_pred) + eps)
    noise_sdr = (n * n_pred).sum() / (torch.linalg.norm(n) * torch.linalg.norm(n_pred) + eps)
    a = torch.sum(y ** 2) / (torch.sum(y ** 2) + torch.sum(n ** 2) + eps)
    return -(a * y_sdr + (1 - a) * noise_sdr)


def train_epoch(dataloader, model, device, optimizer, accumulation_steps):
    model.train()
    sum_loss = 0
    crit_l1 = nn.L1Loss()

    for itr, (X_batch, y_batch) in enumerate(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        mask = model(X_batch)
        loss = crit_l1(mask * X_batch, y_batch) / accumulation_steps
        loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        sum_loss += loss.item() * len(X_batch)

    if (itr + 1) % accumulation_steps != 0:
        optimizer.step()
        model.zero_grad()

    return sum_loss / len(dataloader.dataset)


def validate_epoch(dataloader, model, device):
    model.eval()
    sum_loss = 0
    crit_l1 = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            y_batch = spec_utils.crop_center(y_batch, y_pred)
            loss = crit_l1(y_pred, y_batch)
            sum_loss += loss.item() * len(X_batch)

    return sum_loss / len(dataloader.dataset)


def load_val_filelist(val_filelist_path):
    if val_filelist_path:
        with open(val_filelist_path, 'r', encoding='utf8') as f:
            return json.load(f)
    return []


def split_dataset(args, logger, timestamp):
    val_filelist = load_val_filelist(args.val_filelist)

    # Attempt to split dataset
    result = dataset.train_val_split(
        dataset_dir=args.dataset,
        split_mode=args.split_mode,
        val_rate=args.val_rate,
        val_filelist=val_filelist
    )
    
    # Check if the result is valid
    if result is None or not isinstance(result, tuple) or len(result) != 2:
        logger.error("Error in train_val_split: Expected a tuple of (train_filelist, val_filelist).")
        raise TypeError("train_val_split did not return a valid result.")

    train_filelist, val_filelist = result

    # Additional debug handling
    if args.debug:
        logger.info('### DEBUG MODE')
        train_filelist, val_filelist = train_filelist[:1], val_filelist[:1]

    elif args.val_filelist is None and args.split_mode == 'random':
        with open(f'val_{timestamp}.json', 'w', encoding='utf8') as f:
            json.dump(val_filelist, f, ensure_ascii=False)

    for i, (X_fname, y_fname) in enumerate(val_filelist):
        logger.info(f'{i + 1} {os.path.basename(X_fname)} {os.path.basename(y_fname)}')

    return train_filelist, val_filelist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=2019)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--dataset', '-d', required=True)
    p.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    p.add_argument('--learning_rate', '-l', type=float, default=0.001)
    p.add_argument('--lr_min', type=float, default=0.0001)
    p.add_argument('--lr_decay_factor', type=float, default=0.9)
    p.add_argument('--lr_decay_patience', type=int, default=6)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--accumulation_steps', '-A', type=int, default=1)
    p.add_argument('--cropsize', '-C', type=int, default=256)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--val_rate', '-v', type=float, default=0.2)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=4)
    p.add_argument('--val_cropsize', '-c', type=int, default=256)
    p.add_argument('--num_workers', '-w', type=int, default=4)
    p.add_argument('--epoch', '-E', type=int, default=200)
    p.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    p.add_argument('--reduction_level', '-L', type=float, default=0.2)
    p.add_argument('--mixup_rate', '-M', type=float, default=0.0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    logger = setup_logger(__name__, f'train_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_filelist, val_filelist = split_dataset(args, logger, timestamp=datetime.now().strftime("%Y%m%d%H%M%S"))

    # Model, optimizer, and scheduler setup
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    model = netta.CascadedNet(args.n_fft, args.hop_length, 32, 128).to(device)
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.lr_min, verbose=True
    )

    # Data loading
    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset.VocalRemoverTrainingSet(
            filelist=train_filelist * args.patches, cropsize=args.cropsize, reduction_rate=args.reduction_rate
        ),
        batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=dataset.VocalRemoverValidationSet(
            patch_list=dataset.make_validation_set(
                filelist=val_filelist, cropsize=args.val_cropsize, sr=args.sr, hop_length=args.hop_length, n_fft=args.n_fft
            )
        ),
        batch_size=args.val_batchsize, shuffle=False, num_workers=args.num_workers
    )

    # Training loop
    best_loss = np.inf
    log = []

    for epoch in range(args.epoch):
        logger.info(f'# epoch {epoch}')
        train_loss = train_epoch(train_dataloader, model, device, optimizer, args.accumulation_steps)
        val_loss = validate_epoch(val_dataloader, model, device)
        logger.info(f'  * training loss = {train_loss:.6f}, validation loss = {val_loss:.6f}')

        scheduler.step(val_loss)
        log.append([train_loss, val_loss])

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'model_{datetime.now().strftime("%Y%m%d%H%M%S")}.pth')

        if optimizer.param_groups[0]['lr'] <= args.lr_min:
            break


if __name__ == '__main__':
    main()

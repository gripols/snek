import argparse

import chainer
from chainer import backends
import librosa
import numpy as np
from tqdm import tqdm

from lib import spec_utils
from lib import unet
# FIXME
"""
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
"""


def load_model(args):
    model = unet.SpecUNet()
    chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()
    return model


def load_audio(input_path, sr=44100):
    print('Loading raw waveform...', end=' ')
    audio, _ = librosa.load(input_path, sr, mono=False, dtype=np.float32)
    print('done')
    return audio


def process_audio(model, audio, args):
    print('Mixture STFT...', end=' ')
    X, phase = spec_utils.calc_spec(audio, True)
    print('done')

    ref_max = X.max()
    X /= ref_max

    left = model.offset
    roi_size = args.cropsize - left * 2
    right = roi_size + left - (X.shape[2] % left)
    X_pad = np.pad(X, ((0, 0), (0, 0), (left, right)), mode='edge')

    inst_preds, vocal_preds = [], []
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for j in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
            start = j * roi_size
            X_window = X_pad[None, :, :, start:start + args.cropsize]
            X_tta = np.concatenate([
                X_window,
                X_window[:, :, :, ::-1],
                X_window[:, ::-1, :, :],
                X_window[:, ::-1, :, ::-1],
            ])
            mask = model(model.xp.asarray(X_tta))
            mask = backends.cuda.to_cpu(mask.data)
            mask[1] = mask[1, :, :, ::-1]
            mask[2] = mask[2, ::-1, :, :]
            mask[3] = mask[3, ::-1, :, ::-1]
            mask = mask.mean(axis=0)[None]
            X_window = spec_utils.crop_and_concat(mask, X_window, False)
            inst_preds.append((X_window * mask)[0])
            vocal_preds.append((X_window * (1 - mask))[0])

    return inst_preds, vocal_preds, phase, ref_max


def save_audio(waveform, filename, sr=44100):
    # do I even want .wav files at this point
    librosa.output.write_wav(filename, waveform, sr)


def main():
    args = parse_arguments()

    model = load_model(args)
    audio = load_audio(args.input, args.sr)
    inst_preds, vocal_preds, phase, ref_max = process_audio(model, audio, args)

    inst_preds = np.concatenate(inst_preds, axis=2)
    print('Instrumental inverse STFT...', end=' ')
    instrumental_waveform = spec_utils.spec_to_wav(
        inst_preds, phase, 512, ref_max)
    print('done')
    save_audio(instrumental_waveform, 'instrumental.wav', args.sr)

    vocal_preds = np.concatenate(vocal_preds, axis=2)
    print('Vocal inverse STFT...', end=' ')
    vocal_waveform = spec_utils.spec_to_wav(vocal_preds, phase, 512, ref_max)
    print('done')
    save_audio(vocal_waveform, 'vocal.wav', args.sr)


if __name__ == '__main__':
    main()

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display

from lib import dataset
from lib import netta
from lib import spec_utils
from lib import utils


class Separator(object):

    def __init__(self, model, device=None, batchsize=1, cropsize=256, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _postprocess(self, X_spec, mask):
        if self.postprocess:
            mask_mag = np.abs(mask)
            mask_mag = spec_utils.merge_artifacts(mask_mag)
            mask = mask_mag * np.exp(1.j * np.angle(mask))

        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        # y_spec = X_spec * mask
        # v_spec = X_spec - y_spec

        return y_spec, v_spec

    def _separate(self, X_spec_pad, roi_size):
        X_dataset = []
        patches = (X_spec_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_spec_crop = X_spec_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_spec_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask_list = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                mask = self.model.predict_mask(torch.abs(X_batch))

                mask = mask.detach().cpu().numpy()
                mask = np.concatenate(mask, axis=2)
                mask_list.append(mask)

            mask = np.concatenate(mask_list, axis=2)

        return mask

    def separate(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(
            n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(
            X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= np.abs(X_spec).max()

        mask = self._separate(X_spec_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(X_spec, mask)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(
            n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(
            X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask = self._separate(X_spec_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_spec_pad = np.pad(
            X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask_tta = self._separate(X_spec_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(X_spec, mask)

        return y_spec, v_spec


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.pth')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P',
                   type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--output_dir', '-o', type=str, default="")
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
    model = netta.CascadedNet(args.n_fft, args.hop_length, 32, 128)
    model.load_state_dict(torch.load(
        args.pretrained_model, map_location='cpu'))
    model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast'
    )
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print('done')

    if X.ndim == 1:
        # mono to stereo
        X = np.asarray([X, X])

    print('stft of wave source...', end=' ')
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print('done')

    sp = Separator(
        model=model,
        device=device,
        batchsize=args.batchsize,
        cropsize=args.cropsize,
        postprocess=args.postprocess
    )

    if args.tta:
        y_spec, v_spec = sp.separate_tta(X_spec)
    else:
        y_spec, v_spec = sp.separate(X_spec)

    # Inverse STFT for separated instruments and vocals
    print('Inverse STFT of instruments...', end=' ')
    wave_instruments = spec_utils.spectrogram_to_wave(
        y_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}{}_Instruments.wav'.format(
        output_dir, basename), wave_instruments.T, sr)

    print('Inverse STFT of vocals...', end=' ')
    wave_vocals = spec_utils.spectrogram_to_wave(
        v_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}{}_Vocals.wav'.format(output_dir, basename), wave_vocals.T, sr)

    # Now generate and save spectrograms for instruments
    print('Generating spectrogram for instruments...')
    D_instruments = librosa.stft(wave_instruments)

# Loop through each channel
# D_instruments.shape[0] is 2 for stereo
    for i in range(D_instruments.shape[0]):
        S_db_instruments = librosa.amplitude_to_db(
            np.abs(D_instruments[i]), ref=np.max)
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(S_db_instruments, sr=sr,
                                 x_axis='time', y_axis='log', cmap='viridis')
        plt.title(f'Spectrogram of Instruments Channel {i + 1} (dB)')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True)
        plt.savefig('{}{}_Instruments_Channel_{}.png'.format(
            output_dir, basename, i + 1), dpi=300, bbox_inches='tight')
        plt.close()

    # Now generate and save spectrograms for vocals
    print('Generating spectrogram for vocals...')
    D_vocals = librosa.stft(wave_vocals)

    # Loop through each channel
    for i in range(D_vocals.shape[0]):  # D_vocals.shape[0] is 2 for stereo
        S_db_vocals = librosa.amplitude_to_db(np.abs(D_vocals[i]), ref=np.max)
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(
            S_db_vocals, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.title(f'Spectrogram of Vocals Channel {i + 1} (dB)')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True)
        plt.savefig('{}{}_Vocals_Channel_{}.png'.format(
            output_dir, basename, i + 1), dpi=300, bbox_inches='tight')
        plt.close()

    if args.output_image:
        # Optional: if you still want to save the original images as well
        image = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite('{}{}_Instruments.jpg'.format(
            output_dir, basename), image)

        image = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite('{}{}_Vocals.jpg'.format(output_dir, basename), image)


if __name__ == '__main__':
    main()

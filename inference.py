import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import netta
from lib import spec_utils
from lib import utils

class Separator:
    # good
    def __init__(self, model, device=None, batchsize=1, cropsize=256, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    # post process audio spectra
    def _postprocess(self, X_spec, mask):
        if self.postprocess:
            mask_mag = np.abs(mask)
            mask_mag = spec_utils.merge_artifacts(mask_mag)
            mask = mask_mag * np.exp(1.j * np.angle(mask))

        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

        return y_spec, v_spec


    def _separate(self, X_spec_pad, roi_size):
        patches = (X_spec_pad.shape[2] - 2 * self.offset) // roi_size
        X_dataset = [
            X_spec_pad[:, :, i * roi_size: (i + 1)] for i in range(patches)
        ]
        X_dataset = np.asarray(X_dataset)
        

        self.model.eval()
        with torch.no_grad():
            mask_list = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = torch.from_numpy(X_dataset[i: i + self.batchsize]).to(self.device) 
                mask = self.model.predict_mask(torch.abs(X_batch))
                mask_list.append(mask.detach().cpu().numpy())

            mask = np.concatenate(mask_list, axis=2)

        return mask


    def separate(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= np.abs(X_spec).max()

        mask = self._separate(X_spec_pad, roi_size)
        mask = mask[:, :, :n_frame]

        return self._postprocess(X_spec, mask) 


    def separate_tta(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask = self._separate(X_spec_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask_tta = self._separate(X_spec_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta) * 0.5

        return self._postprocess(X_spec, mask) 

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.pth')

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default=DEFAULT_MODEL_PATH)
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

def init_device(gpu):
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f'cude:{gpu}')
    return torch.device('cpu')


def load_model(model_path, n_fft, hop_length, device):
    model = netta.CascadedNet(n_fft, hop_length, 32, 128)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    return model


def load_audio(input_path, sr):
    print('Loading audio source...', end=' ')
    X, _ = librosa.load(input_path, sr=sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
    if X.ndim == 1:
        X = np.asarray([X, X])  # mono to stereo
    print('done')
    return X


def save_results(output_dir, basename, y_spec, v_spec, sr, output_image):
    # TODO: improve spectrogram images 
    os.makedirs(output_dir, exist_ok=True)

    sf.write(os.path.join(output_dir, f'{basename}_Instruments.wav'), spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length).T, sr)
    sf.write(os.path.join(output_dir, f'{basename}_Vocals.wav'), spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length).T, sr)

    if output_image:
        image_y = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite(os.path.join(output_dir, f'{basename}_Instruments.jpg'), image_y)

        image_v = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite(os.path.join(output_dir, f'{basename}_Vocals.jpg'), image_v)


def main():
    args = parse_arguments()

    print('Loading model', end=' ')
    device = initialize_device(args.gpu)
    model = load_model(args.pretrained_model, args.n_fft, args.hop_length, device)
    print('done')

    X = load_audio(args.input, args.sr)

    print('Calculating STFT of audio source', end=' ')
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print('done')

    separator = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

    if args.tta:
        y_spec, v_spec = separator.separate_tta(X_spec)
    else:
        y_spec, v_spec = separator.separate(X_spec)

    output_dir = args.output_dir.rstrip('/') + '/'
    basename = os.path.splitext(os.path.basename(args.input))[0]

    print('Saving results')
    save_results(output_dir, basename, y_spec, v_spec, args.sr, args.output_image)
    print('Done')


if __name__ == '__main__':
    main()
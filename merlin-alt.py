import argparse
import os
import librosa
import numpy as np
import soundfile as sf
import torch
from plyer import notification  # Import the notification library

# TODO: Merlin Alt or Merlin 2?? That is the question
from lib import dataset
from lib import netta
from lib import spec_utils
import inference


def main():
    # Set up argument parser for command line arguments
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
    
    # Parse the command line arguments
    args = p.parse_args()

    # Load the model
    print('loading model...', end=' ')
    device = torch.device('cpu')  # Default device is CPU
    model = nets.CascadedNet(args.n_fft, args.hop_length)  # Instantiate the model
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))  # Load pretrained weights
    
    # Check if GPU is available and if specified
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))  # Switch to specified GPU
        model.to(device)
    print('done')

    # Create file list from mixtures and instruments
    filelist = dataset.make_pair(args.mixtures, args.instruments)
    
    # Process each mixture-instrument pair
    for mix_path, inst_path in filelist:
        # Optional check to skip specific pairs (commented out)
        # if '_mixture' in mix_path and '_inst' in inst_path:
        #     continue
        # else:
        #     pass

        basename = os.path.splitext(os.path.basename(mix_path))[0]  # Get the base name of the mixture file
        print(basename)

        # Load the wave sources
        print('loading wave source...', end=' ')
        X, sr = librosa.load(mix_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        y, sr = librosa.load(inst_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        print('done')

        # Convert mono to stereo if necessary
        if X.ndim == 1:
            X = np.asarray([X, X])  # Duplicate the mono signal to create a stereo signal

        # Compute the Short-Time Fourier Transform (STFT)
        print('stft of wave source...', end=' ')
        X, y = spec_utils.align_wave_head_and_tail(X, y, sr)  # Align waveforms
        X = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)  # Convert wave to spectrogram
        y = spec_utils.wave_to_spectrogram(y, args.hop_length, args.n_fft)  # Convert instrument wave to spectrogram
        print('done')

        # Initialize the separator and perform separation
        sp = inference.Separator(model, device, args.batchsize, args.cropsize, args.postprocess)
        a_spec, _ = sp.separate_tta(X - y)  # Separate the pseudo instruments

        # Perform inverse STFT to get the pseudo instruments
        print('inverse stft of pseudo instruments...', end=' ')
        pseudo_inst = y + a_spec  # Combine original instrument signal with separated component
        print('done')

        # Save the pseudo instruments as WAV and NumPy files
        sf.write('pseudo/{}_PseudoInstruments.wav'.format(basename), pseudo_inst, sr)  # Save as WAV
        np.save('pseudo/{}_PseudoInstruments.npy'.format(basename), pseudo_inst)  # Save as NumPy array

    # Send system notification after processing is complete
    notification.notify(
        title='Processing Complete',
        message='All audio files have been processed successfully.',
        app_name='Audio Processing Script',
        timeout=10,  # Notification stays for 10 seconds
        # Uncomment the next line and set the path to a custom sound file if desired
        # sound='path_to_custom_sound.wav'  
    )


# Entry point of the script
if __name__ == '__main__':
    main()

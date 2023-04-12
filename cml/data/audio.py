# -*- coding: utf-8 -*-

#import pyo
import librosa
import numpy as np
import os as os
import librosa
from os import path
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Callable, List

#%%
"""
###################
Dataset import helper function
###################
"""
def import_dataset(
        data_path,
        dataset_name,
        split = "train",
        batch_size = 16
        ):
    if (dataset_name in ["gtzan"]):
        return AudioSupervisedDataset(
            dataset_name,
            data_destination = data_path,
            split = split,
            batch_size = batch_size,
            shuffle = split == "train"
        )
    elif (dataset_name == "musclefish"):
        return AudioSupervisedDataset(
            dataset_name = dataset_name,
            data_path = data_path,
            split = split,
            batch_size = batch_size,
            shuffle = split == "train"
        )
        
    
#%%
"""
###################
Dataset class for waveforms
###################
"""  

def simple_audio_preprocess(name, sr=44100, N=44100):
    x, sr = librosa.load(name, sr=sr)
    pad = (N - (len(x) % N)) % N
    x = np.pad(x, (0, int(pad)))
    x = x.reshape(-1, N)
    return x[0]

class AudioSupervisedDataset:
    
    def __init__(self, 
                 dataset_name = None,
                 data_path = None,
                 data_destination = None,
                 split = 'train',
                 batch_size = 16,
                 shuffle = True,
                 preprocess_function: Callable = None,
                 transforms: List[object] = None,
                 extension: str = "*.wav,*.aif",
                 split_percent: float = .2
                 ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.dataset = None
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
        self.extension = extension
        self.ext_list = [s[1:] for s in extension.split(',')]
        self.load_dataset()
        self._reset_generator()
        
    def load_dataset(self):
        if self.data_path is None:
            # Use TFDS loading system
            self.dataset = tfds.load(
                name=self.dataset_name, 
                split=self.split, 
                shuffle_files=self.shuffle, 
                batch_size=self.batch_size)
            self.num_examples = self.info.splits[self.split].num_examples
            self.num_classes = self.info.features['label'].num_classes
        else:
            # Use local dataset
            self.full_path = self.data_path + '/' + self.dataset_name
            self.load_local()
            
    def load_local(self):
        audio_files = []
        # Populate the file list
        for f, folder in enumerate(self.full_path.split(",")):
            print("Recursive search in {}".format(folder))
            for ext in self.extension:
                audio_files.extend(list(Path(folder).rglob(ext)))
        # Extract labels
        audio_files = [str(f) for f in audio_files if os.path.isfile(f)]
        audio_files = [f for f in audio_files if os.path.splitext(f)[1] in self.ext_list]
        labels = [f.split('/')[-2] for f in audio_files]
        labels_names = list(set(labels))
        labels = [labels_names.index(l) for l in labels]
        audio = [simple_audio_preprocess(f) for f in audio_files]
        # Create TF dataset
        self.dataset = tf.data.Dataset.from_tensor_slices({'audio':audio, 'label':labels})
        # Shuffle dataset
        if self.shuffle:
            self.dataset = self.dataset.shuffle(len(audio_files))
        # Generate batches
        self.audio_files = audio_files
        self.labels = labels
        self.labels_names = labels_names
        self.num_examples = len(audio_files)
        self.num_classes = len(self.labels_names)
        self.audio = audio
        self.sr = 44100
    
    def __len__(self):
        return self.num_examples // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = next(self.generator)
        if self.transforms:
            batch = self.transforms(batch)
        return batch
    
    def _reset_generator(self):
        self.generator = self._generate()
    
    def _generate(self):
        ds = self.dataset
        if self.shuffle:
            ds = ds.shuffle(self.num_examples, seed=0)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        for batch in ds:
            audio, labels = batch['audio'], batch['label']
            yield audio, labels
    
    def apply_to_dataset(self, function):
        ds = self.dataset
        if self.split == 'train':
            ds = ds.take(self.num_examples // self.batch_size * self.batch_size)
        ds = ds.batch(self.batch_size)
        ds = ds.map(function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(1)
        results = []
        for batch in ds:
            audio, labels = batch['audio'], batch['label']
            results.append((audio, labels))
        return results
    
    def transform(self, index, name="cqt"):
        return compute_transform(self.audio[index], name, self.sr)
    
    def feature(self, index, name="centroid"):
        return compute_feature(self.audio[index], name)
            
    def apply_transform(self, transform_func):
        if self.dataset is None:
            raise ValueError("Dataset not loaded yet!")
        self.dataset = self.dataset.map(lambda x, y: (transform_func(x), y))
        
    def apply_to_all(self, transform_func):
        if self.dataset is None:
            raise ValueError("Dataset not loaded yet!")
        self.dataset = self.dataset.map(lambda x, y: (transform_func(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    def __getitem__(self, index):
        if self.dataset is None:
            raise ValueError("Dataset not loaded yet!")
        return self.dataset.__getitem__(index)
    
    def __len__(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded yet!")
        return len(self.dataset)

"""
###################
Simple dataloader around tensor
###################
"""     
class WrapperDataset(Dataset):
    def __init__(self, tensor, transforms = None):
        super().__init__()
        self.tensor = tensor
        self.len = self.tensor.shape[0]
        self.transforms = transforms

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.tensor[index]
        if self.transforms is not None:
            data = self.transforms(data)
        return data
 
#%%
"""
###################
Feature computation functions
###################
"""
#
# Main transforms computation function
#
def compute_transform(cur_signal, name, sr=44100, verbose = False):
    # Overall settings
    fSize = 1024
    wSize = fSize
    hSize = fSize//4
    refSr = 44100
    # Constant-Q settings
    fMin = librosa.note_to_hz('C2')
    nBins = 60 * 2
    import warnings
    warnings.filterwarnings('ignore')
    # Perform an analysis of spectral transform for each
    if (name == "stft"):
        # Compute the FFT 
        psc = librosa.stft(cur_signal, n_fft=fSize, win_length=wSize, hop_length=hSize, window='blackman')
        powerspec, phasespec = librosa.magphase(psc);
        return powerspec[:(fSize//2), :]
    elif (name == "mel"):
        # Compute the mel spectrogram        
        wMel = librosa.feature.melspectrogram(cur_signal, sr=sr, n_fft=fSize, hop_length=hSize)
        return wMel
    elif (name == "chroma"):
        # Compute the chromagram
        psc = librosa.stft(cur_signal, n_fft=fSize, win_length=wSize, hop_length=hSize, window='blackman')
        powerspec, phasespec = librosa.magphase(psc);
        wChroma = librosa.feature.chroma_stft(S=powerspec**2, sr=sr)
        return (wChroma)
    elif (name == "cqt"):
        # Compute the Constant-Q transform
        Xcq = librosa.cqt(cur_signal, sr=sr, n_bins=nBins, fmin=fMin, bins_per_octave=12 * 2)
        return (np.abs(Xcq));
    else:
        raise NotImplementedError("Audio descriptor " + name + " not implemented")

#
# Main features computation function
#
def compute_feature(cur_signal, name, verbose = False):
    # Window sizes
    wSize = 1024
    if (name == "loudness"):
        return librosa.feature.rms(y = cur_signal)
    # Compute the spectral centroid. [y, sr, S, n_fft, ...]
    elif name == "centroid":
        return librosa.feature.spectral_centroid(y = cur_signal)
        # Compute the spectral bandwidth. [y, sr, S, n_fft, ...]
    elif name == "bandwidth":
        return (librosa.feature.spectral_bandwidth(y = cur_signal))
        # Compute spectral contrast [R16] , sr, S, n_fft, ...])	
    elif name == "contrast":
        return (librosa.feature.spectral_contrast(y = cur_signal))
        # Compute the spectral flatness. [y, sr, S, n_fft, ...]
    elif name == "flatness":
        return (librosa.feature.spectral_flatness(y = cur_signal))
        # Compute roll-off frequency
    elif name == "rolloff":
        return (librosa.feature.spectral_rolloff(y = cur_signal))
    else:
        raise NotImplementedError("Audio descriptor " + name + " not implemented")


#%%
"""
###################

Generative waveform dataset

###################
"""
def start_pyo_server(sr):
    # Server parameters
    channels = 1
    # Start server
    s = pyo.Server(audio="offline")
    # set server parameters
    s.setSamplingRate(sr)
    s.setNchnls(channels)
    s.setVerbosity(1)
    s.boot()
    return s

"""
Dataset 1 - FM-synthesisa + Moog filter:
1. Carrier frequency j (3 carriers)
2. Ratio k 
3. Index r
4. Filter resonance Q
5. Filter frequency f
"""
def generate_fm_set(n_samples=1e5, duration = 0.075, slice_id = None, midi_range=[20, 80], sr=22050):    
    target_file = '/tmp/toy_fm_' + str(n_samples) + '_' + str(duration) + '.npy'
    if (path.exists(target_file)):
        return np.load(target_file)
    # Server settings
    s = start_pyo_server(sr)
    fformat = 'WAVE'
    sampletype = 0  
    # Subroutine to generate one sound
    def generate_fm_sound():
        # Randomly sample parameters
        f0 = librosa.midi_to_hz(np.random.randint(midi_range[0], midi_range[1]))
        # Harmonic factor
        harm = np.arange(1, 5, 0.5)[np.random.randint(0, 6)]
        # Carrier frequencies
        freq2 = f0 + harm * f0 / 10
        freq3 = f0 - harm * f0 / 10
        freq_ar = [f0, freq2, freq3]
        # Ratio k
        k = np.arange(0.1, 0.4, 0.05)[np.random.randint(0, 6)]
        ratio_ar = [k, k, k]
        # Index r
        r = np.arange(0, 10, 1)[np.random.randint(0, 8)]
        # Filter resonance
        Q = np.arange(1, 11, 1)[np.random.randint(0, 10)]
        # Filter frequency
        filter_f = np.arange(1, 11, 1)[np.random.randint(0, 10)]
        # Set recording
        s.recordOptions(dur=(duration + 0.1), filename="/tmp/gen.wav", fileformat=fformat, sampletype=sampletype)
        # Create sound through FM
        sig = pyo.FM(carrier=freq_ar, ratio=ratio_ar, index=float(r), mul=1)
        # Filter corresponding sound with Moog Low Pass
        sig_filtered = pyo.MoogLP(sig, freq=float(filter_f) * 22050 / 200, res=float(Q), mul=1).out()
        # start the render
        s.start()
        # cleanup
        s.recstop()
    # Generate dataset
    dataset = []
    for s_id in range(int(n_samples)):
        generate_fm_sound()
        y, sr = librosa.load('/tmp/gen.wav')
        y = y[int(0.1 * sr):]
        if (slice_id is not None):
            y = y[slice_id[0]:slice_id[1]]
        dataset.append(y[np.newaxis, :])
    s.shutdown()
    data = np.concatenate(dataset)
    np.save(target_file, data)
    return data

def generate_additive_set(n_samples=1e5, duration = 0.075, slice_id = None, midi_range=[20, 50], sr=22050):
    """
    Dataset 2 - Additive synthesis with BiQuad filter:
    1. Harmonicity = sum(s3.bn/in((2n+j)*f*i), j = [0,1,0.1]
    2. # Partials k
    3. Filter (Q)
    4. Filter (f)
    """
    target_file = '/tmp/toy_add_' + str(n_samples) + '_' + str(duration) + '.npy'
    if (path.exists(target_file)):
        return np.load(target_file)
    # Server settings
    s = start_pyo_server(sr)
    fformat = 'WAVE'
    sampletype = 0  
    # Subroutine to generate one sound
    def generate_additive_sound():
        # Randomly sample parameters
        f0 = librosa.midi_to_hz(np.random.randint(midi_range[0], midi_range[1]))
        freq_ar = [f0]
        # Number of harmonics
        n_harm = np.arange(0, 20, 1)[np.random.randint(0, 20)]
        # Inharmonicity factor
        inharm = np.arange(0, 0.1, 0.001)[np.random.randint(0, 100)]
        for k in range(n_harm):
            freq_new = (2 * (k + 1 + inharm)) * f0
            if (freq_new < 11025):
                freq_ar.append(freq_new)
        # Filter resonance
        Q = np.arange(1, 11, 1)[np.random.randint(0, 10)]
        # Filter frequency
        filter_f = np.arange(1, 11, 1)[np.random.randint(0, 10)]
        # Set recording
        s.recordOptions(dur=(duration + 0.1), filename="/tmp/gen.wav", fileformat=fformat, sampletype=sampletype)
        # Create sound
        sin_sig = pyo.Sine(freq=freq_ar, mul=1)
        # Filter with biquad
        sig_filtered = pyo.Biquad(sin_sig, freq=float(filter_f * 22050 / (2 * 100)), q=float(Q), type=2).out()
        # start the render
        s.start()
        # cleanup
        s.recstop()
    # Generate dataset
    dataset = []
    for s_id in range(int(n_samples)):
        generate_additive_sound()
        y, sr = librosa.load('/tmp/gen.wav')
        y = y[int(0.1 * sr):]
        if (slice_id is not None):
            shift = int(np.random.rand() * 100)
            y = y[slice_id[0]+shift:slice_id[1]+shift]
        y = y / np.max(y)
        dataset.append(y[np.newaxis, :])
    s.shutdown()
    data = np.concatenate(dataset)
    np.save(target_file, data)
    return data



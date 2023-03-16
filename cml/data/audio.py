# -*- coding: utf-8 -*-

import pyo
import librosa
import numpy as np
from os import path
import librosa
import matplotlib.pyplot as plt
import numpy as np
from os import path
from pathlib import Path
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
# Lightning specific import
import pytorch_lightning as pl
# Import transforms
from transforms import PitchShift, Reverb, Noise, Dequantize, Reverse
    
#%%
"""
###################
Dataloader for waveforms
###################
"""  
def dummy_load(name, args):
    """
    Preprocess function that takes one audio path and load it into
    chunks of 2048 samples.
    """
    x = librosa.load(name, args.sr)[0]
    if (x.shape[0] > (args.shape)):
        m_point = x.shape[0] - (args.shape)
        r_point = int(np.random.random() * m_point)
        x = x[r_point:r_point+int(args.shape)]
    else:
        x = np.concatenate([x, np.zeros(int(args.shape - int(x.shape[0])))])
    return x
    
class WaveformDataset(Dataset):
    def __init__(self, 
                 data_dir = None, 
                 preprocess_function = dummy_load, 
                 transforms = None, 
                 extension = "*.wav,*.aif", 
                 split_percent = .2, 
                 split_set = "train",
                 args = None):
        super().__init__()
        # Check we have data dir
        assert data_dir is not None
        self.folder_list = data_dir
        self.preprocess_function = preprocess_function
        self.extension = extension
        self.transforms = transforms
        self.args = args
        # Import the data
        self._preprocess()
        if self.len == 0:
            raise Exception("No data found !")
        # Create list of indexes
        self.index = np.arange(self.len)
        np.random.shuffle(self.index)
        if split_set == "train":
            self.len = int(np.floor((1 - split_percent) * self.len))
            self.offset = 0
        elif split_set == "test":
            self.offset = int(np.floor((1 - split_percent) * self.len))
            self.len = self.len - self.offset
        elif split_set == "full":
            self.offset = 0

    def _preprocess(self):
        extension = self.extension.split(",")
        idx = 0
        wavs = []
        # Populate the file list
        if self.folder_list is not None:
            for f, folder in enumerate(self.folder_list.split(",")):
                print("Recursive search in {}".format(folder))
                for ext in extension:
                    wavs.extend(list(Path(folder).rglob(ext)))
        else:
            with open(self.file_list, "r") as file_list:
                wavs = file_list.read().split("\n")
        self.env = [None] * len(wavs);
        loader = tqdm(wavs)
        for wav in loader:
            loader.set_description("{}".format(path.basename(wav)))
            output = self.preprocess_function(wav, self.args)
            if output is not None:
                self.env[idx] = output
                idx += 1
        self.len = len(self.env)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.env[self.index[index + self.offset]]
        if self.transforms is not None:
            data = self.transforms(data)
        return data

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.io import wavfile 
import audioread 
import librosa

ROOT_DIR = 'cats_dogs/'
CSV_PATH = 'train_test_split.csv'

def load_audio_data():
    """
    Reads audio files and returns the time series, sampling rates, channel count, and wav file names
    """
    df = pd.read_csv(CSV_PATH)
    file_names, time_series, sampling_rates, channels = [], [] ,[], []
    durations = []

    for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
        v = list(df[k].dropna())
        
        for f in v:
            file_names.append(f)
            # Read and get data and sampling rate of audio
            ts, sr = librosa.load(ROOT_DIR + f,sr=16000)
            time_series.append(ts)
            sampling_rates.append(sr)
            
            # Calculate duration of each file
            duration = len(ts) / sr 
            durations.append(duration)

            # Count number of channels within audio 
            with audioread.audio_open(ROOT_DIR + f) as input_file:
                channels.append(input_file.channels)
        
    return file_names, time_series, sampling_rates, channels, durations

def extract_mel_features(time_series):
    freq, mfccs, delta_mfcc = [],[],[]
    
    for ts in time_series:
        # Store frequencies
        fr = librosa.feature.melspectrogram(y=ts,sr=16000)
        freq.append(fr)
        # delta_mel.append(librosa.feature.delta(fr))
        
        # Store MFCCs and corresponding deltas     
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(fr),sr=1600)
        mfccs.append(mfcc)
        delta_mfcc.append(librosa.feature.delta(mfcc))

    return freq, mfccs, delta_mfcc

names, ts, sr, channels, durations = load_audio_data()
fr, mfccs, mfcc_deltas = extract_mel_features(ts)
audio = pd.DataFrame({'file':names, 'data':ts, 'sample rate':sr,'channel count':channels, 'duration':durations, 'frequency':fr, 'mfccs':mfccs, 'delta mfcc':mfcc_deltas})
print(audio.head())

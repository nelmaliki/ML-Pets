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
            file_names.append(ROOT_DIR+f)
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
    freq, mfccs, delta_mfcc, spectral = [],[],[],[]
    for i in range(20):
        mfccs.append([])
        delta_mfcc.append([])

    zcrs = []

    for ts in time_series:
        # Store frequencies
        fr = librosa.feature.melspectrogram(y=ts,sr=16000)
        freq.append(fr)
        # delta_mel.append(librosa.feature.delta(fr))
        
        # Store MFCCs and corresponding deltas     
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(fr),sr=16000)
        j=0
        for coef in mfcc:
            mfccs[j].append(np.mean(coef))
            j += 1

        delta = librosa.feature.delta(mfcc)
        j=0
        for coef in delta:
            delta_mfcc[j].append(np.mean(coef))
            j += 1

        #Spectral Centroid
        spectral.append(np.mean(librosa.feature.spectral_centroid(ts, sr=16000)[0]))

        #Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(ts)
        zcrs.append(zcr)

    return freq, mfccs, delta_mfcc, spectral, zcrs

#Adding the class label to the dataframe
def add_class(row):
    name = row[len(ROOT_DIR): ] 
    label = None
    if 'cat' in name:
        label = 'cat'
    elif 'dog' in name:
        label = 'dog'
    return label

def get_dataframe():
    names, ts, sr, channels, durations = load_audio_data()
    fr, mfccs, mfcc_deltas, spectral, zcrs = extract_mel_features(ts)
    i = 0
    audio = pd.DataFrame({'file':names, 'data':ts, 'sample rate':sr,'channel count':channels, 'duration':durations, 'frequency':fr, 'spectral':spectral, 'zcrs': zcrs})
    for coef in mfccs:
        audio[("mfccs_" + str(i))] = coef
        i += 1
    i = 0
    for coef in mfcc_deltas:
        audio[("mfcc_deltas_" + str(i))] = coef
        i += 1
    print(audio.head())
    audio['label'] = audio.apply(lambda x: add_class(x['file']),axis=1)
    audio['label'] = audio['label'].astype('category')
    audio['label'] = audio['label'].cat.codes # Turn to binary
    return audio
    
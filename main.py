import utils
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import librosa

sound_csv = "noise_numbers.csv"


def main():
    df = None
    #Sound_csv file is a csv with the processed audio data
    if os.path.isfile(sound_csv):
        df = pd.read_csv(sound_csv)
    #If we dont have that data we will have to generate it
    else:
        print("Csv file of processed wav files was not found, this might take awhile...")
        # Utils.py will process data in cats_dogs according to train_test_split.csv
        df = utils.load_dataset()
        df.to_csv(sound_csv)

    #Testing file access from df
    test, sr = librosa.load(df.loc[4,"file"])
    
    #Fast Fourier Transformation
    X = np.fft.fft(test)

    #Lets graph this thing
    X_mag = np.absolute(X)
    plt.figure(figsize=(18, 5))
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag))
    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('Frequency (Hz)')
    plt.title("title")
    plt.show()

if __name__ == "__main__":
    main()
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import librosa
import sklearn as sk
import librosa.display


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
        print(pd.read_csv('train_test_split.csv'))
        df = utils.load_dataset()
        df.to_csv(sound_csv)

    #Testing file access from df
    test, sr = librosa.load(df.loc[4,"file"])

    #Getting Spectral_Centroids
    spectral_centroids = librosa.feature.spectral_centroid(test, sr=sr)[0]
    print(spectral_centroids.shape)  # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    # Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(test, sr=sr, alpha=0.4)
    plt.plot(t, sk.preprocessing.minmax_scale(spectral_centroids, axis=0), color='r')
    #plt.show()
    
    #Mel Cepstral
    mfccs = librosa.feature.mfcc(test, sr=sr)
    print(mfccs.shape)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    #plt.show()


    #Lets try fitting a model with this stuff
    df["type"].replace(to_replace=["Dog", "Cat"], value=[0.0, 1.0], inplace=True)
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
        df.drop(["type"], axis=1), df[["type"]], test_size=0.3)



    train_spectral_centroids = np.asarray([np.mean(librosa.feature.spectral_centroid(test, sr=sr)[0]) for test, sr in [librosa.load(file) for file in x_train["file"]]]).reshape(-1,1)

    test_spectral_centroids = np.asarray([np.mean(librosa.feature.spectral_centroid(test, sr=sr)[0]) for test, sr in [librosa.load(file) for file in x_test["file"]]]).reshape(-1,1)


    model = sk.linear_model.LogisticRegression().fit(train_spectral_centroids,y_train)
    pred = model.predict(test_spectral_centroids)
    print_stats(y_test, pred)

    #Todo: Lasso to determine useful Mel coefficients

def print_stats(y_test, pred):
    cm = sk.metrics.confusion_matrix(y_test, np.round(pred), labels=[0, 1])
    print("Confusion Matrix:\n ", cm)
    print("\nTest Accuracy = ", str(sk.metrics.accuracy_score(y_test, np.round(pred))))
    print("\nError rate = ", str(1-sk.metrics.accuracy_score(y_test, np.round(pred))))
    tn, fp, fn, tp = cm.ravel()
    if (tp+fp != 0):
        print("\nTrue Positive = ", tp / (tp + fp))
    else:
        print("True positive NaN")
    if (tp + fn != 0):
        print("\nTrue Negative = ", tn / (tn + fn))
    else:
        print("True Negative NaN")


if __name__ == "__main__":
    main()
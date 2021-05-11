import utils as utils
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import librosa
import sklearn as sk
import librosa.display
import sklearn.ensemble

sound_csv = "noise_numbers.csv"


def main():
    df = utils.get_dataframe()

    #Testing file access from df
    test, sr = librosa.load(df.loc[4,"file"])

    # Check albels are correct
    print(df["label"])
    
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
    predictor = ["spectral"]
    
    predictor.extend(["mfccs_" + str(i) for i in range(20)])
    predictor.extend(["mfcc_deltas_" + str(i) for i in range(20)])

    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
       df[predictor], df["label"].values, test_size=0.3)
        

    model = sk.linear_model.LogisticRegression().fit(x_train,y_train)
    pred = model.predict(x_test)
    print_stats(y_test, pred)


    #Lots of predictors so prune pretty aggressively
    model = sk.linear_model.Lasso(.67).fit(x_train,y_train)
    pred = model.predict(x_test)
    print_stats(y_test, pred)
    best_predictors = []
    for coef_ind in range(len(model.coef_)):
        if model.coef_[coef_ind] != 0:
            best_predictors.append(predictor[coef_ind])
    print("Best predictors = ", best_predictors)

    # Try clustering
    X = df[predictor]
    pca = sk.decomposition.PCA(n_components = 2, svd_solver='full')
    pca.fit(X)
    Xa = pca.transform(X)
    plt.plot(Xa[:,0],Xa[:,1],'.')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

    #Random Forest Classifier
    score_list = []
    best_score = 0
    for i in range (1,21):
        model = sk.ensemble.RandomForestClassifier(n_estimators=i*10)
        fit = model.fit(x_train, y_train)
        pred = model.predict(x_test)
        score = sk.metrics.accuracy_score(y_test, np.round(pred))
        score_list.append(score)
        best_score = max(best_score, score)

    print("Best score: ", best_score)
    plt.plot([i*10 for i in range(1,21)], score_list)
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy Score")
    plt.title("Random Forest Classifier n_estimators")
    plt.show()


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
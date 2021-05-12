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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import sklearn.model_selection


def main():
    df = utils.get_dataframe()
    kfold = sk.model_selection.KFold(shuffle=True)
    print(df["file"])
    #CAT VIZ
    test, sr = librosa.load("cats_dogs/cat_10.wav")
    #Getting Spectral_Centroids
    spectral_centroids = librosa.feature.spectral_centroid(test, sr=sr)[0]
    print(spectral_centroids.shape)  # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    model_accuracy_list = []
    model_f1_list = []

    # Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(test, sr=sr, alpha=0.4)
    plt.plot(t, sk.preprocessing.minmax_scale(spectral_centroids, axis=0), color='r')
    plt.legend(["minimax scaled centroids", "Waveforms"])
    plt.xlabel("Frame")
    plt.ylabel("Frequency")
    plt.title("Spectral Centroids of a Cat")
    plt.show()
    
    #Mel Cepstral
    mfccs = librosa.feature.mfcc(test, sr=sr)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.ylabel("Coefficients")
    plt.yscale("linear")
    plt.title("Mel Cepstral Visualization of a Cat")
    plt.show()

    # DOG VIZ
    test, sr = librosa.load("cats_dogs/dog_barking_93.wav")
    # Getting Spectral_Centroids
    spectral_centroids = librosa.feature.spectral_centroid(test, sr=sr)[0]
    print(spectral_centroids.shape)  # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    # Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(test, sr=sr, alpha=0.4)
    plt.plot(t, sk.preprocessing.minmax_scale(spectral_centroids, axis=0), color='r')
    plt.legend(["minimax scaled centroids", "Waveforms"])
    plt.xlabel("Frame")
    plt.ylabel("Frequency")
    plt.title("Spectral Centroids of a Dog")
    plt.show()

    # Mel Cepstral
    mfccs = librosa.feature.mfcc(test, sr=sr)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.ylabel("Coefficients")
    plt.yscale("linear")
    plt.title("Mel Cepstral Visualization of a Dog")
    plt.show()


    #Lets try fitting a model with this stuff
    predictor = ["spectral"]
    
    predictor.extend(["mfccs_" + str(i) for i in range(20)])
    predictor.extend(["mfcc_deltas_" + str(i) for i in range(20)])
    x = df[predictor]
    y = df["label"].values
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
       x, y, test_size=0.3)



    #Lots of predictors so prune pretty aggressively


    best_predictors = []
    best_score = 0
    best_f1_score = 0
    score_list = []
    for i in range(0,100):
        #This is basically lasso but C = 1-alpha
        model = sk.linear_model.LogisticRegression(penalty="l1",C=1-i/100, solver="liblinear")
        f1_score = np.mean(sk.model_selection.cross_val_score(model, x, y, cv=kfold,
                                                           scoring=sk.metrics.make_scorer(sk.metrics.f1_score)))
        score = np.mean(sk.model_selection.cross_val_score(model, x, y, cv=kfold,
                                                              scoring=sk.metrics.make_scorer(sk.metrics.accuracy_score)))
        score_list.append(score)
        model.fit(x,y)
        best_f1_score = max(f1_score, best_f1_score)
        if score >= best_score:
            best_score = score
            best_predictors = []
            coef = model.coef_[0]
            for coef_ind in range(len(coef)):
                if coef[coef_ind] != 0:
                    best_predictors.append(predictor[coef_ind])



    model_f1_list.append(best_f1_score)
    model_accuracy_list.append(best_score)


    plt.plot([i/100 for i in range(0,100)], score_list)
    plt.xlabel("Alpha")
    plt.ylabel("f1 score")
    plt.title("Determining Optimal Alpha value for Lasso")
    plt.show()



    #USING SVM

    X = df[predictor]

    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        X, y, test_size=0.25, random_state=42)



    # defining parameter range
    param_grid = {'C': np.linspace(0.1, 10, 30),
                  'gamma': np.linspace(0.1, 10, 30),
                  'kernel': ['rbf']}

    grid = GridSearchCV(sk.svm.SVC(), param_grid, refit=True)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    model_f1_list.append(np.mean(sk.model_selection.cross_val_score(model, x, y, cv=kfold,
                                                              scoring=sk.metrics.make_scorer(sk.metrics.f1_score))))
    model_accuracy_list.append(np.mean(sk.model_selection.cross_val_score(model, x, y, cv=kfold,
                                                              scoring=sk.metrics.make_scorer(sk.metrics.accuracy_score))))

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)


    #PCA

    """data2 = np.array(data)

    pca = Pipeline([('scaler', StandardScaler()),
                    ('clf', sk.decomposition.PCA(n_components=2, svd_solver='full', random_state=42))])
    pca.fit(data2)
    Xa = pca.transform(data2)
    plt.scatter(Xa[:, 0], Xa[:, 1], color=[assign_color(x) for x in new_y], marker='.', s=5)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Clustering MFCCs and MFCC deltas (Standard Scaler)')
    plt.show()

    pca = Pipeline([('scaler', MinMaxScaler()), ('clf', sk.decomposition.PCA(n_components = 2, svd_solver='full', random_state = 42))])
    pca.fit(data2)
    Xa = pca.transform(data2)
    plt.scatter(Xa[:,0], Xa[:,1],color=[assign_color(x) for x in new_y],marker='.',s=5)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Clustering MFCCs and MFCC deltas (MinMax Scaler)')
    plt.show()"""


    #Random Forest Classifier

    score_list_all_predictors = []
    best_score = 0
    for i in range(1, 11):
        model = sk.ensemble.RandomForestClassifier(n_estimators=i * 10)
        score = np.mean(sk.model_selection.cross_val_score(model,x,y,cv=kfold,scoring=sk.metrics.make_scorer(sk.metrics.accuracy_score)))
        score_list_all_predictors.append(score)
        best_score = max(best_score, score)

    score_list = []
    best_score = 0
    best_x = x[best_predictors]
    best_f1_score = 0
    for i in range (1,11):
        model = sk.ensemble.RandomForestClassifier(n_estimators=i*10)
        score = np.mean(sk.model_selection.cross_val_score(model,best_x,y,cv=kfold,scoring=sk.metrics.make_scorer(sk.metrics.accuracy_score)))
        f1_score = np.mean(sk.model_selection.cross_val_score(model,best_x,y,cv=kfold,scoring=sk.metrics.make_scorer(sk.metrics.f1_score, zero_division=1)))
        best_f1_score = max(f1_score, best_f1_score)
        score_list.append(score)
        best_score = max(best_score, score)

    model_accuracy_list.append(best_score)
    model_f1_list.append(best_f1_score)

    plt.plot([i*10 for i in range(1,11)], score_list)
    plt.plot([i * 10 for i in range(1, 11)], score_list_all_predictors)
    plt.legend(["'best' predictors", "all predictors"])
    plt.xlabel("n_estimators")
    plt.ylabel("f1 Score")
    plt.title("Random Forest Classifier n_estimators")
    plt.show()

    #PRint comparison between models:

    print("Accuracy", model_accuracy_list)
    print("F1 Scores", model_f1_list)
    
    models = ['Lasso', 'SVM', 'Random Forests']
    plt.bar(models, model_accuracy_list)
    plt.xlabel("Model Type")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.show()


    plt.bar(models, model_f1_list)
    plt.xlabel("Model Type")
    plt.ylabel("F1 Score")
    plt.title("Model F1 Comparison")
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

def assign_color(x):
    if x == 0:
        return "C0"
    else:
        return "C1"

if __name__ == "__main__":
    main()
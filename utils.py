import numpy as np  # linear algebra
import pandas as pd  # CSV file
import scipy.io.wavfile as sci_wav  # Open wav files
import random
import librosa

ROOT_DIR = 'cats_dogs/'
CSV_PATH = 'train_test_split.csv'

def load_dataset():
    '''Load the dataset in a dictionary.
    From the dataframe, it reads the [train_cat, train_dog, test_cat, test_dog]
    columns and loads their corresponding arrays into the <dataset> dictionary

    Params:
        dataframe: a pandas dataframe with 4 columns [train_cat, train_dog,
        test_cat, test_dog]. In each columns, many WAV names (eg. ['cat_1.wav',
        'cat_2.wav']) which are going to be read and append into a list

    Return:
        dataset = {
            'train_cat': [[0,2,3,6,1,4,8,...],[2,5,4,6,8,7,4,5,...],...]
            'train_dog': [[sound 1],[sound 2],...]
            'test_cat': [[sound 1],[sound 2],...]
            'test_dog': [[sound 1],[sound 2],...]
        }
    '''
    df = pd.read_csv(CSV_PATH)


    rows = []
    for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
        v = list(df[k].dropna())
        type = None
        if k == 'train_cat' or k=="test_cat":
            type = "Cat"
        elif k == 'train_dog' or k=="test_dog":
            type = "Dog"
        for f in v:
            row = {"type": type, "file": ROOT_DIR+f}
            rows.append(row)


    dataset = pd.DataFrame(rows)
    return dataset



df = pd.read_csv(CSV_PATH)

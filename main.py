import utils
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
sound_csv = "noise_numbers.csv"
def main():
    df = None
    if os.path.isfile(sound_csv):
        df = pd.read_csv(sound_csv)
    else:
        print("Csv file of processed wav files was not found, this might take awhilez...")
        data = utils.load_dataset()
        noise = []
        for i in range(0,max([len(l) for l in list(data.values())])):
            tmp = []
            for l in list(data.values()):
                if i >= len(l):
                    tmp.append(None)
                else:
                    tmp.append(l[i])
            noise.append(tmp)
        df = pd.DataFrame(noise, columns = list(data.keys()))
        print("Dataframe created, saving to ", sound_csv)
        df.to_csv(sound_csv)

    print(df)
    sns.heatmap(np.abs(df.corr()), annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap")
    plt.show()
if __name__ == "__main__":
    main()
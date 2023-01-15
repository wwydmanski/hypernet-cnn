import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def load_uci_data(DATA):
    if DATA == "Ionosphere":
        ionosphere = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data', header=None)
        ionosphere = ionosphere.drop(1, axis=1)
        X = ionosphere.values[:, :-1].astype(float)
        y = ionosphere.values[:, -1]
        y = LabelEncoder().fit_transform(y).astype(int)
        #(351, 33) 2 #auc roc
        GS_METRIC = "balanced_accuracy"

    if DATA == "Libras":
        libras = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data', header=None)
        X = libras.values[:, :-1].astype(float)
        y = libras.values[:, -1].astype(int)
        y -= 1
        #(360, 90) 15
        GS_METRIC = "balanced_accuracy"

    if DATA == "Lymphography":
        lymphography = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data', header=None)

        lymphography = lymphography[lymphography[0] != 4]
        lymphography = lymphography[lymphography[0] != 1]
        lymphography[0] = lymphography[0] - 1

        X = lymphography.values[:, 1:].astype(float)
        y = lymphography.values[:, 0].astype(int)
        y -= 1

        #(148, 18) 4
        GS_METRIC = "balanced_accuracy"

    #     X, y = imblearn.over_sampling.SMOTE(
    #                         sampling_strategy={0:10, 3:20}, 
    #                         random_state=42,
    #                         k_neighbors=1,
    #                     ).fit_resample(X, y)


    
    
    
    
    print(X.shape, len(np.unique(y)))
    
    return X, y

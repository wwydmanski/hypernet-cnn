import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_data(dataset_name):
    DATA = dataset_name
    
    if DATA == "Ionosphere":
        ionosphere = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data', header=None)
        ionosphere = ionosphere.drop(1, axis=1)
        X = ionosphere.values[:, :-1].astype(float)
        y = ionosphere.values[:, -1]
        y = LabelEncoder().fit_transform(y).astype(int)
        #(351, 33) 2
        GS_METRIC = "balanced_accuracy"
        
    elif DATA == "Libras":
        libras = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data', header=None)
        X = libras.values[:, :-1].astype(float)
        y = libras.values[:, -1].astype(int)
        y = LabelEncoder().fit_transform(y).astype(int)
        #(360, 90) 15
        GS_METRIC = "balanced_accuracy"
    
    elif DATA == "Lymphography":
        lymphography = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data', header=None)
        X = lymphography.values[:, 1:].astype(float)
        y = lymphography.values[:, 0].astype(int)
        y = LabelEncoder().fit_transform(y).astype(int)
        #(148, 18) 4
        GS_METRIC = "balanced_accuracy"
        
    elif DATA == "Hill-Valley-without-noise":
        hill_valley_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_without_noise_Training.data')
        hill_valley_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_without_noise_Testing.data')

        X_train = hill_valley_train.values[:, :-1].astype(float)
        y_train = hill_valley_train.values[:, -1]
        y_train = LabelEncoder().fit_transform(y_train).astype(int)

        X_test = hill_valley_test.values[:, :-1].astype(float)
        y_test = hill_valley_test.values[:, -1]
        y_test = LabelEncoder().fit_transform(y_test).astype(int)

        print('train', X_train.shape, len(np.unique(y_train)))
        print('test', X_test.shape, len(np.unique(y_test)))

        X = (X_train, X_test)
        y = (y_train, y_test)
        
    elif DATA == "Hill-Valley-with-noise":
        hill_valley_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_with_noise_Training.data')
        hill_valley_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_with_noise_Testing.data')

        X_train = hill_valley_train.values[:, :-1].astype(float)
        y_train = hill_valley_train.values[:, -1]
        y_train = LabelEncoder().fit_transform(y_train).astype(int)

        X_test = hill_valley_test.values[:, :-1].astype(float)
        y_test = hill_valley_test.values[:, -1]
        y_test = LabelEncoder().fit_transform(y_test).astype(int)

        print('train', X_train.shape, len(np.unique(y_train)))
        print('test', X_test.shape, len(np.unique(y_test)))

        X = (X_train, X_test)
        y = (y_train, y_test)
        
    elif DATA == "cnae":
        cnae = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data', header=None)
        cnae = cnae.drop(1, axis=1)
        X = cnae.values[:, 1:].astype(float)
        y = cnae.values[:, 0]
        y = LabelEncoder().fit_transform(y).astype(int)
        #(1080, 855) 2
        print(X.shape, len(np.unique(y)))
        
    
    return X, y
        
        
        
        
        
        
    



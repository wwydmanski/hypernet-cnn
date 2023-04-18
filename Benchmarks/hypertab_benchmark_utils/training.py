import imblearn
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from collections import defaultdict
from tqdm import trange
import seaborn as sns
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sklearn
import time
import datetime
import random
import torch
import pyhopper
import seaborn as sns
import os
import pandas as pd

GS_METRIC="balanced_accuracy"

def is_test_run():
    TEST_RUN = os.environ.get("HYPERTAB_TEST_RUN", 'False')
    if TEST_RUN == 'True' or TEST_RUN == 'true' or TEST_RUN == '1':
        return True
    return False
        

def get_n_classes(X, y):
    n_classes = len(np.unique(y if not isinstance(y, tuple) else y[0]))
    print('n_classes', n_classes)
    return n_classes


def get_n_features(X, y):
    n_features = X.shape[1] if not isinstance(X, tuple) else X[0].shape[1]
    print('n_features', n_features)
    return n_features


def get_each_class_counts(X, y):
    unique, counts = np.unique(y if not isinstance(y, tuple) else np.array(y), return_counts=True)
    res = dict(zip(unique, counts))
    print('class counts', res)
    return res


def train_test_split_tuple(X, y, train_size=None):
    if isinstance(X, tuple) and isinstance(y, tuple):
        X_train, X_test = X
        y_train, y_test = y
    else:    
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y)
    
    return X_train, X_test, y_train, y_test


def print_mean_std_max(res_df, dataset_name, metric=GS_METRIC):
    print('metric', metric)
    print('dataset_name', dataset_name)
    res = res_df[res_df["Class"]==metric].reset_index(drop=True)["Metric"]
    print(f"{dataset_name}: {res.mean():.2f} ~ {res.std():.2f} (max: {res.max():.2f})") 
    
    
def prepare_data(X, y, size=None):
    if isinstance(X, tuple) and isinstance(y, tuple):
        X_train, X_test = X
        y_train, y_test = y
    else:    
        X_train, X_test, y_train, y_test = train_test_split_tuple(X, y, train_size=size)
    # X_train, y_train = imblearn.over_sampling.RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train, X_test, y_train, y_test = [torch.from_numpy(x) for x in [X_train, X_test, y_train, y_test]]
    
    return X_train, X_test, y_train, y_test


def _summarize_results(y_pred, y_score, y_test, labels):
    results = []
    for idx, label in enumerate(labels):
        y_pred_filt = y_pred[y_test==idx]
        y_test_filt = y_test[y_test==idx]
        acc = (y_pred_filt==y_test_filt.numpy()).sum()/len(y_test_filt)*100
        results.append({
            "Class": label,
            "Metric": acc
        })
        
    acc = (y_pred==y_test.numpy()).sum()/len(y_test)*100    
    results.append({
        "Class": "Total",
        "Metric": acc
    })
    
    
    results.append({
        "Class": "balanced_accuracy",
        "Metric": balanced_accuracy_score(y_test, torch.from_numpy(y_pred)).item()*100
    })
    
    try:
        results.append({
            "Class": "F1 score",
            "Metric": f1_score(y_test, torch.from_numpy(y_pred)).item()*100
        })
        results.append({
            "Class": "roc_auc",
            "Metric": roc_auc_score(y_test, torch.from_numpy(y_score[:, 1])).item()*100
        })
        results.append({
            "Class": "Precision",
            "Metric": precision_score(y_test, torch.from_numpy(y_pred)).item()*100
        })
        results.append({
            "Class": "Recall",
            "Metric": recall_score(y_test, torch.from_numpy(y_pred)).item()*100
        })
    except ValueError:
        pass
    return results


def test_model(*, model_fn, data, train_size, label_encoder=None, iters=10, as_numpy=False): 
    if is_test_run():
        iters = 1
        
    if label_encoder is not None:
        labels = label_encoder.classes_
    else:
        labels = sorted(pd.unique(data[1][0] if isinstance(data[1], tuple) else data[1]))

    
    results = []

    for i in range(iters):
        X_train, X_test, y_train, y_test = prepare_data(*data, train_size)
        print('iter', i+1, 'of', iters, 'X_train shape', X_train.shape)

        model = model_fn()

        if as_numpy:
            model.fit(X_train.numpy(), y_train.numpy());
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        results.extend(_summarize_results(y_pred, y_score, y_test, labels))

    dframe = pd.DataFrame.from_dict(results)
#     sns.violinplot(data=dframe[dframe["Class"]!="Loss"], y="Class", x="Metric", orient='h')
    return dframe


def pyhopper_best_params(*, model_fn, param_grid, data, train_size, DATA, metric=GS_METRIC, time="30m", default_params={}, device='cuda:0'):
    if is_test_run():
        print("THIS IS TEST RUN")
        time = 60
        if 'epochs' in param_grid:
            param_grid["epochs"] = pyhopper.choice([10])
    
    X, y = data
    print('| DEVICE:', 'cpu' if not device else device)
    print('| model_fn', model_fn.__name__)
    print()
    print('pyhopper', 'X.shape:', X.shape, 'y.shape:', y.shape, 'train_size:', train_size)
        
    def objective(params):
    #     print("Training...")
        print('params',params)
        
        model_results = test_model(
            model_fn=model_fn(
                **default_params,
                **params
            ),
            data=(X, y),
            train_size=train_size,
            iters=5
        )
        
        with open(f"{DATA}_{model_fn.__name__}_params.txt", "a") as f:
            f.write(str(params) + ", " + str(model_results[model_results["Class"]==metric]["Metric"].mean()) + "\n")
        return model_results[model_results["Class"]==metric]["Metric"].mean()

    from pyhopper.callbacks import History
    search = pyhopper.Search(param_grid)

    best_params = search.run(objective, "maximize", time, n_jobs="1x per-gpu" if 'cuda' in device else 1, seeding_ratio=0.5)
    
    with open(f"{DATA}_{model_fn.__name__}_best_params.txt", "a") as f:
            f.write(str(best_params))
    
    print(f"{DATA}_{model_fn.__name__}_{best_params}")
    return best_params, search.history


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    
def initial_split(X, y, benchmark_data_size=0.8):
    X_train, X_test, y_train, y_test = train_test_split_tuple(X, y, train_size=benchmark_data_size)
    return X_train, X_test, y_train, y_test
    

def get_eval_and_benchmark_size(*, X_train, train_size=0.6, valid_size=0.2):
    _benchmark_size_frac = train_size / (valid_size + train_size)
    
    train_and_valid_max_size = int(len(X_train))
    train_max_size = int(len(X_train) * _benchmark_size_frac)
    
    print('eval_max_size', train_and_valid_max_size)
    print('train_max_size', train_max_size)
    
    return train_and_valid_max_size, train_max_size
    
    
    
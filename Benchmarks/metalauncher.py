import papermill
import multiprocessing

DATASETS = [
    "BreastCancer",
    "Connectionist",
    "Dermatology",
    "Glass",
    "Cleveland",
]

DEVICES = ["cuda:1", "cuda:3"]

def run(dataset):
    device = DEVICES.pop()
    papermill.execute_notebook(
        "UCI-3.ipynb",
        "UCI-3_{}.ipynb".format(dataset),
        parameters=dict(DATA=dataset, DEVICE=device),
    )
    DEVICES.append(device)

with multiprocessing.Pool(2) as pool:
    pool.map(run, DATASETS)
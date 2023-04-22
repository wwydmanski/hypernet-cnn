import papermill
import multiprocessing

DATASETS = [
    "Ionosphere",
    "Libras",
    "Lymphography",
    "Hill-Valley-without-noise",
    "Hill-Valley-with-noise",
    "cnae"
]

DEVICES = ["cuda:0"]

def run(dataset):
    print(dataset)
    device = DEVICES.pop()
    papermill.execute_notebook(
        "UCI-3-GPU.ipynb",
        "UCI-3-GPU_{}.ipynb".format(dataset),
        parameters=dict(DATA=dataset, DEVICE=device, TEST_RUN=True),
    )
    DEVICES.append(device)

with multiprocessing.Pool(1) as pool:
    pool.map(run, DATASETS)

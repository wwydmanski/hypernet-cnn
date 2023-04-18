import node
from qhoptim.pyt import QHAdam

from sklearn.ensemble import RandomForestClassifier
import torch
from tabular_hypernet import HypernetworkPCA, TrainingModes, Hypernetwork
from tabular_hypernet.modules import SimpleNetwork
from tabular_hypernet.interfaces import HypernetworkSklearnInterface, SimpleSklearnInterface
from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
 
# In case of bagging and ensemble:
#   first_hidden_layer == hypertab.mask_size
#   second_hidden_layer == hypertab.target_size
def get_parametrized_bagged_fn():    
    def bagged_fn(first_hidden_layer, second_hidden_layer, batch_size, max_iter, learning_rate_init):
        def _inner():
            model = BaggingClassifier(
                estimator=MLPClassifier(
                    hidden_layer_sizes=(first_hidden_layer, second_hidden_layer), 
                    max_iter=max_iter, 
                    batch_size=batch_size,
                    learning_rate_init=learning_rate_init,
                    verbose=False
                )
            )
            return model
        return _inner
    return bagged_fn

# n_models == hypertab.mask_no
def get_parametrized_ensemble_fn():    
    def ensemble_fn(n_models, first_hidden_layer, second_hidden_layer, batch_size, max_iter, learning_rate_init):
        def _inner():
            model = VotingClassifier(
                estimators=[
                    (
                        str(i), 
                        MLPClassifier(
                            hidden_layer_sizes=(first_hidden_layer, second_hidden_layer), 
                            max_iter=max_iter, 
                            batch_size=batch_size,
                            learning_rate_init=learning_rate_init,
                            verbose=False
                        )
                    ) for i in range(n_models)
                ],
                voting='soft'
            )
            return model
        return _inner
    return ensemble_fn


def get_parametrized_xgboost_fn(*, seed):
    def get_xgboost(**params):
        random_seed = seed
        def _inner(**args):
            return XGBClassifier(
                verbosity=0,
                random_state=random_seed,
                use_label_encoder=False,
                **params,
                **args
            )
        return _inner    
    return get_xgboost

    
def get_parametrized_node_fn(*, X_train, n_classes, n_features, DEVICE):
    def node_fn(layer_dim=128, num_layers=1, depth=3, batch_size=32):
        def _inner():
            network = torch.nn.Sequential(
                node.DenseBlock(n_features, 
                                layer_dim=layer_dim,
                                num_layers=num_layers, 
                                tree_dim=n_classes+1, 
                                depth=depth, 
                                flatten_output=False,
                                choice_function=node.entmax15, 
                                bin_function=node.entmoid15
                               ),
                node.Lambda(lambda x: x.mean(dim=1))
            )


            network = network.to(DEVICE)
            network.device=DEVICE

            with torch.no_grad():
                res = network(torch.as_tensor(X_train[:1000], device=DEVICE).to(torch.float32))


            optimizer_params = { 'nus':(0.7, 1.0), 'betas':(0.95, 0.998) }
            optim = QHAdam(network.parameters(), **optimizer_params)

            network = SimpleSklearnInterface(network, device=DEVICE, epochs=150, batch_size=batch_size)
            network.optimizer = optim
            return network
        return _inner
    return node_fn


def get_parametrized_dropout_net1(*, DEVICE, n_features, n_classes):
    def network_fn1(epochs=100, drop1=0.3, drop2=0.5, batch_size=32, lr=3e-4):
        def _inner():
            network = torch.nn.Sequential(
                            torch.nn.Dropout(drop1),
                            torch.nn.Linear(n_features, 64),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(drop2),
                            torch.nn.Linear(64, n_classes)
                        ).to(DEVICE).train()

            network = SimpleSklearnInterface(network, epochs=epochs, batch_size=batch_size, lr=lr, device=DEVICE)
            return network
        return _inner
    return network_fn1


def get_parametrized_dropout_net2(*, DEVICE, n_features, n_classes):
    def network_fn2(epochs=100, drop1=0.3, drop2=0.5, drop3=0.5, batch_size=32, lr=3e-4):
        def _inner():
            network = torch.nn.Sequential(
                            torch.nn.Dropout(drop1),
                            torch.nn.Linear(n_features, 64),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(drop2),
                            torch.nn.Linear(64, 64),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(drop3),
                            torch.nn.Linear(64, n_classes)
                        ).to(DEVICE).train()

            network = SimpleSklearnInterface(network, epochs=epochs, batch_size=batch_size, lr=lr, device=DEVICE)
            return network
        return _inner
    return network_fn2


def get_parametrized_dropout_net3(*, DEVICE, n_features, n_classes):
    def network_fn3(epochs=100, drop1=0.3, drop2=0.5, drop3=0.5, drop4=0.5, batch_size=32, lr=3e-4):
        def _inner():
            network = torch.nn.Sequential(
                            torch.nn.Dropout(drop1),
                            torch.nn.Linear(n_features, 64),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(drop2),
                            torch.nn.Linear(64, 128),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(drop3),
                            torch.nn.Linear(128, 64),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(drop4),
                            torch.nn.Linear(64, n_classes)
                        ).to(DEVICE).train()

            network = SimpleSklearnInterface(network, epochs=epochs, batch_size=batch_size, lr=lr, device=DEVICE)
            return network
        return _inner
    return network_fn3


def get_parametrized_hypertab_pca_fn(*, DEVICE, n_classes):
    def network_pca_fn(epochs=100, masks_no=100, mask_size=100, target_size=100, n_comp=5, lr=3e-4, batch_size=64, verbose=False):
        def _inner():
            hypernet = HypernetworkPCA(
                            target_architecture=[(mask_size, target_size), (target_size, n_classes)], 
                            test_nodes=masks_no,
                            architecture=torch.nn.Sequential(torch.nn.Linear(n_comp, 64), 
                                torch.nn.ReLU(),
                                torch.nn.Linear(64, 128),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(),
                                torch.nn.Linear(128, 128),
                                torch.nn.ReLU(),
                            ),
                            mode=TrainingModes.CARTHESIAN,
                            input_size=n_features
                        ).to(DEVICE)    
            hypernet = hypernet.train()

            network = HypernetworkSklearnInterface(hypernet, device=DEVICE, epochs=epochs, batch_size=batch_size, verbose=verbose, lr=lr)
            return network
        return _inner
    return network_pca_fn


def get_parametrized_hypertab_fn(*, DEVICE, n_classes, n_features):
    def network_hp_fn(epochs=150, masks_no=100, mask_size=100, target_size=100, lr=3e-4, batch_size=64, verbose=False):
        def _inner():
            hypernet = Hypernetwork(
                            target_architecture=[(mask_size, target_size), (target_size, n_classes)],
                            test_nodes=masks_no,
                            architecture=torch.nn.Sequential(torch.nn.Linear(n_features, 64), 
                                torch.nn.ReLU(),
                                torch.nn.Linear(64, 128),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(),
                                torch.nn.Linear(128, 128),
                                torch.nn.ReLU(),
                            ),
                            mode=TrainingModes.CARTHESIAN,
                        ).to(DEVICE)    
            hypernet = hypernet.train()

            network = HypernetworkSklearnInterface(hypernet, device=DEVICE, epochs=epochs, batch_size=batch_size, verbose=verbose, lr=lr)
            return network
        return _inner
    return network_hp_fn


def get_parametrized_rf_fn(*, seed):
    def get_rf(**params):
        random_seed = seed
        def _inner():
            return RandomForestClassifier(
                random_state=random_seed,
                **params
            )
        return _inner
    return get_rf









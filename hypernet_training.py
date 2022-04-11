#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import load_dotenv
load_dotenv()

import os


# In[2]:


from comet_ml import Experiment, Optimizer

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data_utils
import pandas as pd
from collections import defaultdict

torch.set_default_dtype(torch.float32)


# In[3]:


from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


# ## Declare classes

# In[4]:


class SimpleNetwork(torch.nn.Module):
    def __init__(self, inp_size, layers=[100], outputs=10):
        super().__init__()
        self.layers = []
        
        self.inp = torch.nn.Linear(inp_size, layers[0])
        self.output = torch.nn.Linear(layers[0], outputs)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        out = x
        out = self.inp(out)
        out = self.relu(out)
        out = self.output(out)
        return out
    
class DropoutNetwork(torch.nn.Module):
    def __init__(self, inp_size, layers=[100]):
        super().__init__()
        self.layers = []
        
        self.inp = torch.nn.Linear(inp_size, layers[0])
        self.output = torch.nn.Linear(layers[0], 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(1-700/784)
    
    def forward(self, x):
        out = x
        out = self.dropout(out)
        out = self.inp(out)
        out = self.relu(out)
        out = self.output(out)
        return out
    
class InsertableNet(SimpleNetwork):
    def __init__(self, weights, inp_size, out_size, layers=[10]):
        super().__init__(inp_size, layers)
        input_w_size = inp_size*layers[0]
        input_b_size = layers[0]

        hidden_w_size = layers[0]*out_size
        hidden_b_size = out_size

        self.inp_weights = weights[:input_w_size].reshape((layers[0], inp_size))
        self.inp_bias = weights[input_w_size:input_w_size+input_b_size]

        self.output_weights = weights[input_w_size+input_b_size:input_w_size+input_b_size+hidden_w_size].reshape((out_size, layers[0]))
        self.output_bias = weights[input_w_size+input_b_size+hidden_w_size:input_w_size+input_b_size+hidden_w_size+hidden_b_size]
        self.out_act = torch.nn.Sigmoid()
        
    def forward(self, data):
        out = F.linear(data, self.inp_weights, self.inp_bias)
        out = self.relu(out)
        out = F.linear(out, self.output_weights, self.output_bias)
        return out
    
class MaskedNetwork(SimpleNetwork):
    def __init__(self, input_size, mask_size, layers=[10]):
        super().__init__(mask_size, layers=layers)
        template = np.zeros(input_size)
        mask = np.random.choice(len(template), mask_size, False)
        template[mask] = 1
        self.mask = torch.from_numpy(template).to(torch.bool)
        
    def forward(self, x):
        data = x[:, self.mask]
        return super().forward(data)


# In[5]:


class Hypernetwork(torch.nn.Module):
    def __init__(self, inp_size=784, out_size=10, mask_size=20, node_hidden_size=20, layers=[64, 256, 128], test_nodes=100, device='cuda:0'):
        super().__init__()
        self.target_outsize = out_size
        self.device = device
        
        self.mask_size = mask_size
        self.input_size = inp_size
        self.node_hidden_size = node_hidden_size
        
        input_w_size = mask_size*node_hidden_size
        input_b_size = node_hidden_size

        hidden_w_size = node_hidden_size*out_size
        hidden_b_size = out_size
            
        self.out_size = input_w_size+input_b_size+hidden_w_size+hidden_b_size
        
        self.input = torch.nn.Linear(inp_size, layers[0])
        self.hidden1 = torch.nn.Linear(layers[0], layers[1])
        self.hidden2 = torch.nn.Linear(layers[1], layers[2])
        self.out = torch.nn.Linear(layers[2], self.out_size)
        
        self.dropout = torch.nn.Dropout()
        
#         self.relu = torch.nn.ReLU()
        self.relu = torch.relu
        self.template = np.zeros(inp_size)
        self.test_nodes = test_nodes
        self.test_mask = self._create_mask(test_nodes)
        
        self._retrained = True
        self._test_nets = None
        
    def to(self, device):
        super().to(device)
        self.device = device
        self.test_mask = self._create_mask(self.test_nodes)
        return self
        
    def forward(self, data, mask=None):
        """ Get a hypernet prediction. 
        During training we use a single target network per sample. 
        During eval, we create a network for each test mask and average their results
        
        Args:
            data - prediction input
            mask - either None or a torch.tensor((data.shape[0], data.shape[1])).
        """
        if self.training:
            self._retrained = True
            if mask is None:
                masks = np.array([np.random.choice((len(self.template)), self.mask_size, False) for _ in range(len(data))])
                tmp = np.array([self.template.copy() for _ in range(len(data))])
                for i, mask in enumerate(masks):
                    tmp[i, mask] = 1
                mask = torch.from_numpy(tmp).to(torch.float32).to(self.device)

            
            # If we have a few identical masks in a row
            # we only need to calculate target network
            # for the first one
            recalculate = [True]*len(mask)
            for i in range(1, len(mask)):
                if torch.equal(mask[i-1], mask[i]):
                    recalculate[i] = False
                    
            weights = self.craft_network(mask)
            mask = mask.to(torch.bool)
            
            res = torch.zeros((len(data), self.target_outsize)).to(self.device)
            for i in range(len(data)):
                if recalculate[i]:
                    nn = InsertableNet(weights[i], self.mask_size, self.target_outsize, layers=[self.node_hidden_size])
                masked_data = data[i, mask[i]]
                res[i] = nn(masked_data)
            return res
        else:
            if mask is None:
                mask = self.test_mask
                nets = self._get_test_nets()
            else:
                nets = self.__craft_nets(mask)
            mask = mask.to(torch.bool)

            res = torch.zeros((len(data), self.target_outsize)).to(self.device)
            for i in range(len(mask)):
                nn = nets[i]
                masked_data = data[:, mask[i]]
                res += nn(masked_data)
            res /= self.test_nodes
            return res

    def _get_test_nets(self):
        if self._retrained:
            nets = self.__craft_nets(self.test_mask)
            self._test_nets = nets
            self._retrained = False
        return self._test_nets
    
    def __craft_nets(self, mask):
        nets = []
        weights = self.craft_network(mask.to(torch.float32))
        for i in range(len(mask)):
            nn = InsertableNet(weights[i], self.mask_size, self.target_outsize, layers=[self.node_hidden_size])
            nets.append(nn)
        return nets
        
    def _create_mask(self, count):
        masks = np.array([np.random.choice((len(self.template)), self.mask_size, False) for _ in range(count)])
        tmp = np.array([self.template.copy() for _ in range(count)])
        for i, mask in enumerate(masks):
            tmp[i, mask] = 1
        mask = torch.from_numpy(tmp).to(torch.float32).to(self.device)
        return mask
    
    def craft_network(self, mask):
        out = self.input(mask)
        out = self.relu(out)
        
        out = self.hidden1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.hidden2(out)
        out = self.relu(out)

        out = self.out(out)
        return out
    

def get_dataset(size=60000, masked=False, mask_no=200, mask_size=700, shared_mask=False, batch_size=32, test_batch_size=32):
    mods = [transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)),    #mean and std of MNIST
        transforms.Lambda(lambda x: torch.flatten(x))]
    mods = transforms.Compose(mods)
    
    trainset = datasets.MNIST(root='./data/train', train=True, download=True, transform=mods)
    testset = datasets.MNIST(root='./data/test', train=False, download=True, transform=mods)
    if masked:
        trainset = MaskedDataset(trainset, mask_no, mask_size)
        testset = MaskedDataset(testset, mask_no, mask_size)
        if shared_mask:
            testset.masks = trainset.masks
        
    indices = torch.arange(size)
    trainset = data_utils.Subset(trainset, indices)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)
    return trainloader, testloader



class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, masks, mask_size):
        inputs = dataset[0][0].shape[0]
        self.mask_size = mask_size
        self.dataset = dataset
        self.template = np.zeros(inputs)
        self.masks = self._create_mask(masks)
        self.masks_indices = np.random.choice(np.arange(masks), len(dataset))
        
    def _create_mask(self, count):
        masks = np.array([np.random.choice((len(self.template)), self.mask_size, False) for _ in range(count)])
        tmp = np.array([self.template.copy() for _ in range(count)])
        for i, mask in enumerate(masks):
            tmp[i, mask] = 1
        mask = torch.from_numpy(tmp).to(torch.float32)
        return mask
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        mask = self.masks[self.masks_indices[idx]]
        return image, label, mask
    

def train_regular(network, optimizer, criterion, loaders, size, epochs, 
                  device='cuda:0', 
                  name="dense-net", 
                  test_every=10, experiment=None):
    
    if experiment is None:
        experiment = Experiment(api_key=os.environ.get("COMET_KEY"), project_name="hypernetwork")
    experiment.add_tag(name)
    experiment.log_parameter("training_size", size)
    experiment.log_parameter("max_epochs", epochs)
    experiment.log_parameter("check_val_every_n_epoch", 5)
    
    trainloader, testloader = loaders
    train_loss = []
    test_loss = []
    test_accs = []
    with trange(epochs) as t:
        for epoch in t:
            total_loss = 0
            running_loss = 0.0
            correct = 0
            total = 0
            network = network.train()
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                correct += (outputs.argmax(1)==labels).float().sum()
                total += outputs.shape[0]
                running_loss += loss.item()
                train_loss.append(loss.item())
                if i>0 and i % 100 == 0:
                    total_loss += running_loss/100
                    running_loss = 0.0
                    correct = 0
                    total=0

            total_loss = 0
            correct = 0
            denom = 0
            network = network.eval()
            if epoch%test_every==0:
                for i, data in enumerate(testloader):
                    images, labels = data
                    denom += len(labels)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = network(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total_loss += criterion(outputs, labels).item()

                test_loss.append(total_loss/i)
                test_accs.append(correct/denom*100)

                t.set_postfix(test_acc=correct/denom*100, loss=total_loss/i)
                experiment.log_metric("test_accuracy", correct/denom*100, step=epoch)    
                experiment.log_metric("test_loss", test_loss[-1], step=epoch)
                
    experiment.end()
    return max(test_accs), test_loss[-1]


def train_loop(hypernet, optimizer, trainloader, data_size, epochs):
    experiment = Experiment(api_key=os.environ.get("COMET_KEY"), project_name="hypernetwork", display_summary_level=0)
    experiment.add_tag("training")
    experiment.log_parameter("test_nodes", hypernet.test_nodes)
    experiment.log_parameter("mask_size", hypernet.mask_size)
    experiment.log_parameter("training_size", data_size)

    train_loss = []
    test_loss = []
    test_accs = []
    for epoch in range(epochs): 
        total_loss = 0
        running_loss = 0.0
        correct = 0
        total = 0
        hypernet.train()
        for i, data in enumerate(tqdm(trainloader)):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = hypernet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct += (outputs.argmax(1)==labels).float().sum()
            total += outputs.shape[0]
            running_loss += loss.item()
            train_loss.append(loss.item())
            if i>0 and i % 100 == 0:
                total_loss += running_loss/100
                experiment.log_metric("batch_accuracy", correct / total * 100)

                running_loss = 0.0
                correct = 0
                total=0

        total_loss = 0
        correct = 0
        denom = 0
        
        hypernet.eval()
        for i, data in enumerate(testloader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            denom += len(labels)
            
            outputs = hypernet(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_loss += criterion(outputs, labels).item()

        test_loss.append(total_loss/i)
        print(f"Test acc: {correct/denom*100}, loss: {total_loss/i}")
        test_accs.append(correct/denom*100)
        experiment.log_metric("test_accuracy", correct/len(testset)*100, step=epoch)
    experiment.end()    
    return max(test_accs), min(test_loss)


def train_masked(hypernet, optimizer, trainloader, data_size, epochs, masks_no):
    experiment = Experiment(api_key=os.environ.get("COMET_KEY"), project_name="hypernetwork", display_summary_level=0)
    experiment.add_tag("bounded-hypernet")
    experiment.log_parameter("test_nodes", hypernet.test_nodes)
    experiment.log_parameter("mask_size", hypernet.mask_size)
    experiment.log_parameter("training_size", data_size)
    experiment.log_parameter("masks_no", masks_no)

    train_loss = []
    test_loss = []
    for epoch in range(epochs): 
        total_loss = 0
        running_loss = 0.0
        correct = 0
        total = 0
        hypernet.train()
        for i, data in enumerate(tqdm(trainloader)):
            inputs, labels, masks = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()

            outputs = hypernet(inputs, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct += (outputs.argmax(1)==labels).float().sum()
            total += outputs.shape[0]
            running_loss += loss.item()
            train_loss.append(loss.item())
            if i>0 and i % 100 == 0:
                total_loss += running_loss/100
#                 experiment.log_metric("batch_accuracy", correct / total * 100)

                running_loss = 0.0
                correct = 0
                total=0

        total_loss = 0
        correct = 0
        denom = 0
        
        hypernet.eval()
        for i, data in enumerate(testloader):
            images, labels, masks = data
            images = images.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            
            denom += len(labels)
            
            outputs = hypernet(images, masks)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total_loss += criterion(outputs, labels).item()

        test_loss.append(total_loss/i)
        print(f"Test acc: {correct/denom*100}, loss: {total_loss/i}")
        experiment.log_metric("test_accuracy", correct/len(testset)*100, step=epoch)
        experiment.log_metric("test_loss", test_loss[-1], step=epoch)
        
    experiment.end()    
    return correct/len(testset)*100, test_loss[-1]


def train_slow_step(hypernet, optimizer, criterion, loaders, data_size, epochs, masks_no, 
                    experiment=None,
                    tag="slow-step-hypernet", 
                    device='cuda:0', 
                    project_name="hypernetwork",
                    test_every=5):
    """ Train hypernetwork using slow step method - use the same mask for a whole batch, change it once per iteration."""
    if experiment is None:
        experiment = Experiment(api_key=os.environ.get("COMET_KEY"), project_name=project_name, display_summary_level=0)
    experiment.add_tag(tag)
    experiment.log_parameter("test_nodes", hypernet.test_nodes)
    experiment.log_parameter("mask_size", hypernet.mask_size)
    experiment.log_parameter("node_hidden_size", hypernet.node_hidden_size)
    experiment.log_parameter("lr", optimizer.defaults['lr'])
    experiment.log_parameter("training_size", data_size)
    experiment.log_parameter("masks_no", masks_no)
    experiment.log_parameter("max_epochs", epochs)
    experiment.log_parameter("check_val_every_n_epoch", test_every)

    trainloader, testloader = loaders
    train_loss = []
    test_loss = []
    test_accs = []
    mask_idx = 0
    with trange(epochs) as t:
        for epoch in t:
            total_loss = 0
            running_loss = 0.0
            correct = 0
            total = 0
            hypernet.train()
            for i, data in enumerate(trainloader):
                try:
                    inputs, labels, _ = data
                except ValueError:
                    inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                masks = []
                for i in range(len(inputs)):
                    masks.append(hypernet.test_mask[mask_idx])
                masks = torch.stack(masks).to(device)
                mask_idx = (mask_idx+1) % len(hypernet.test_mask)

                optimizer.zero_grad()

                outputs = hypernet(inputs, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                correct += (outputs.argmax(1)==labels).float().sum()
                total += outputs.shape[0]
                running_loss += loss.item()
                train_loss.append(loss.item())
                if i>0 and i % 100 == 0:
                    total_loss += running_loss/100

                    running_loss = 0.0
                    correct = 0
                    total=0

            total_loss = 0
            correct = 0
            denom = 0

            hypernet.eval()
            if epoch%test_every==0:
                for i, data in enumerate(testloader):
                    try:
                        images, labels, _ = data
                    except ValueError:
                        images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    denom += len(labels)

                    outputs = hypernet(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total_loss += criterion(outputs, labels).item()

                test_loss.append(total_loss/i)
                test_accs.append(correct/denom*100)

                t.set_postfix(test_acc=correct/denom*100, loss=total_loss/i)
                experiment.log_metric("test_accuracy", correct/len(testloader.dataset)*100, step=epoch)
                experiment.log_metric("test_loss", test_loss[-1], step=epoch)

    experiment.end()    
    return max(test_accs), test_loss[np.argmax(test_accs)]

def test_model(hypernet, testloader, device='cuda:0', verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    loss = 0
    hypernet.eval()
    for i, data in enumerate(testloader):
        try:
            images, labels, _ = data
        except ValueError:
            images, labels = data
            
        images = images.to(device)
        labels = labels.to(device)
        outputs = hypernet(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        loss += criterion(outputs, labels).item()

    if verbose:
        print(f"Test acc: {correct/len(testloader.dataset)*100:.2f}")
        print(f"Test loss: {total_loss/i}")
    return correct/len(testloader.dataset)*100


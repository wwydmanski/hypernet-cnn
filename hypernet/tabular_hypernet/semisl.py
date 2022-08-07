from dotenv import load_dotenv
load_dotenv()

import os

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

from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


class DatasetUpsampler:
    def __init__(self, dataset, desired_len):
        self.desired_len = desired_len
        self.dataset = dataset
        self.real_len = len(dataset)
        
    def __len__(self):
        return self.desired_len
    
    def __getitem__(self, idx):
        if idx >= self.desired_len:
            raise Error
            
        return self.dataset[idx % self.real_len]
    

def get_dataset(size=(100, 900), mask_no=200, mask_size=700, batch_size=32, test_batch_size=32, shuffle_train=True):
    mods = [transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)),    #mean and std of MNIST
        transforms.Lambda(lambda x: torch.flatten(x))]
    mods = transforms.Compose(mods)
    
    trainset = datasets.MNIST(root='./data/train', train=True, download=True, transform=mods)
    testset = datasets.MNIST(root='./data/test', train=False, download=True, transform=mods)
    
    sup_train_size = size[0]
    unsup_train_size = size[1]
    
    
    ## supervised training dataset
    indices = torch.arange(sup_train_size)
    sup_trainset = data_utils.Subset(trainset, indices)
    
    # balance superivised dataset
    sup_trainset = DatasetUpsampler(sup_trainset, unsup_train_size)
    
    sup_trainloader = torch.utils.data.DataLoader(sup_trainset, batch_size=batch_size,
                                          shuffle=shuffle_train, num_workers=2)
    
    ## unsupervised training dataset
    indices = torch.arange(unsup_train_size) + sup_train_size
    unsup_trainset = data_utils.Subset(trainset, indices)
    
    unsup_trainloader = torch.utils.data.DataLoader(unsup_trainset, batch_size=batch_size,
                                          shuffle=shuffle_train, num_workers=2)
    
    ## test labeled dataset
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)
    
    return sup_trainloader, unsup_trainloader, testloader



class TrainDataLoaderSemi:
    def __init__(self, sup_trainloader, unsup_trainloader):
        self.sup_trainloader = sup_trainloader
        self.unsup_trainloader = unsup_trainloader
        
        if sup_trainloader.batch_size != unsup_trainloader.batch_size:
            raise Error
            
        self.batch_size = sup_trainloader.batch_size
    
    def __len__(self):
        if len(self.sup_trainloader) == len(self.unsup_trainloader):
            return len(self.unsup_trainloader)
        else:
            raise Error
    
    def __iter__(self):
        return zip(self.sup_trainloader, self.unsup_trainloader)
    
    

class TabSSLCrossEntropyLoss(torch.nn.Module):
    def __init__(self, beta=0.1, unsup_target_wrapper=torch.nn.functional.softmax):
        super(TabSSLCrossEntropyLoss, self).__init__()
        
        self.y_f1 = torch.nn.CrossEntropyLoss()
        self.y_f2 = torch.nn.CrossEntropyLoss()
        
        self.f1_f2 = torch.nn.CrossEntropyLoss()
        self.f2_f1 = torch.nn.CrossEntropyLoss()
        
        self.beta = beta
        self.unsup_target_wrapper = unsup_target_wrapper
        self.threshold = None
    
    def forward(self, sup_input, unsup_input):
        sup_outputs1, sup_outputs2, sup_labels = sup_input
        unsup_outputs1, unsup_outputs2 = unsup_input
        
        self.supervised_loss = self.y_f1(sup_outputs1, sup_labels) + self.y_f2(sup_outputs2, sup_labels)
        
        self.self_supervised_loss = self.f1_f2(unsup_outputs1, self.unsup_target_wrapper(unsup_outputs2, dim=1)) \
                                + self.f2_f1(unsup_outputs2, self.unsup_target_wrapper(unsup_outputs1, dim=1))
        
        return self.supervised_loss + self.beta * self.self_supervised_loss
    
    

def train_semisl(hypernet, optimizer, criterion, loaders, data_size, epochs, masks_no,
                    changing_beta=None,
                    log_to_comet=True,
                    experiment=None,
                    tags=["semi-slow-step-hypernet"], 
                    device='cuda:0', 
                    project_name="semi-hypernetwork",
                    test_every=5,
                    description=None,
                    log_params={}):
    """ Train hypernetwork using 2 masks per iteration, one for x1 (sup & unsup), another for x2 (sup & unsup)"""
    trainloader, testloader = loaders
    
    if log_to_comet:
        if experiment is None:
            experiment = Experiment(api_key=os.environ.get("COMET_KEY"), project_name=project_name, display_summary_level=0)
        experiment.add_tags(tags)
        experiment.log_parameter("test_nodes", hypernet.test_nodes)
        experiment.log_parameter("mask_size", hypernet.mask_size)
        experiment.log_parameter("node_hidden_size", hypernet.node_hidden_size)
        experiment.log_parameter("lr", optimizer.defaults['lr'])
        experiment.log_parameter("training_size", sum(data_size))
        experiment.log_parameter("sup_train_size", data_size[0])
        experiment.log_parameter("masks_no", masks_no)
        experiment.log_parameter("max_epochs", epochs)
        experiment.log_parameter("check_val_every_n_epoch", test_every)
        experiment.log_parameter("unsupervised_target_wrapper", criterion.unsup_target_wrapper.__name__)
        experiment.log_parameter("train_batch_size", trainloader.batch_size)
        experiment.log_parameter("test_batch_size", testloader.batch_size)
        experiment.log_parameter("self_sup_loss_threshold", criterion.threshold)
        
        for log_par_k in log_params.keys():
            experiment.log_parameter(log_par_k, log_params[log_par_k])

        
        if description: 
            experiment.log_text(description)      
    
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
            
            supervised_train_loss = 0.
            unsupervised_train_loss = 0.
            train_denom = 0
    
            hypernet.train()
            
            if changing_beta:
                changing_beta(epoch, criterion)
            
            for i, (sup_data, unsup_data) in enumerate(trainloader):
                    
                sup_inputs, sup_labels = sup_data
                unsup_inputs, _ = unsup_data    
                    
                sup_inputs = sup_inputs.to(device)
                sup_labels = sup_labels.to(device)
                unsup_inputs = unsup_inputs.to(device)
                
                ## f1
                masks1 = []
                for i in range(len(sup_inputs)):
                    masks1.append(hypernet.test_mask[mask_idx])
                masks1 = torch.stack(masks1).to(device)
                mask_idx = (mask_idx+1) % len(hypernet.test_mask)
                
                # supervised
                sup_outputs1 = hypernet(sup_inputs, masks1)
                
                # unsupervised
                unsup_outputs1 = hypernet(unsup_inputs, masks1)
        
                ## f2
                masks2 = []
                for i in range(len(sup_inputs)):
                    masks2.append(hypernet.test_mask[mask_idx])
                masks2 = torch.stack(masks2).to(device)
                mask_idx = (mask_idx+1) % len(hypernet.test_mask)
                
                # supervised
                sup_outputs2 = hypernet(sup_inputs, masks2)
                
                # unsupervised
                unsup_outputs2 = hypernet(unsup_inputs, masks2)
                
                
                optimizer.zero_grad()
                
                loss = criterion((sup_outputs1, sup_outputs2, sup_labels), (unsup_outputs1, unsup_outputs2))
                loss.backward()
                optimizer.step()
                

                running_loss += loss.item()
                supervised_train_loss += criterion.supervised_loss
                unsupervised_train_loss += criterion.self_supervised_loss
                train_loss.append(loss.item())
                train_denom += 1
                        
            
            
            if epoch%test_every==0:
                if log_to_comet:
                    experiment.log_metric("beta_coef", criterion.beta, step=epoch)
                    experiment.log_metric('sup_train_loss', supervised_train_loss/train_denom, step=epoch)
                    experiment.log_metric('self_sup_train_loss', unsupervised_train_loss/train_denom, step=epoch)
                    experiment.log_metric('train_loss', running_loss/train_denom, step=epoch)
                
                
                
                # eval
                total_loss = 0
                correct = 0
                denom = 0

                test_criterion = torch.nn.CrossEntropyLoss()
                hypernet.eval()

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
                    total_loss += test_criterion(outputs, labels).item()

                test_loss.append(total_loss/denom)
                test_accs.append(correct/denom*100)

                t.set_postfix(test_acc=correct/denom*100, loss=total_loss/i)
                
                if log_to_comet:
                    experiment.log_metric("test_accuracy", correct/len(testloader.dataset)*100, step=epoch)
                    experiment.log_metric("test_loss", test_loss[-1], step=epoch)
    
    if experiment:
        experiment.end()
                                 
    return max(test_accs), test_loss[np.argmax(test_accs)]
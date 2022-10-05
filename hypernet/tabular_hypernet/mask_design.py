import torch
import numpy as np

def feature_variances(data):
    trainset, _ = data
    stacked_set = torch.stack([x for x, _ in trainset])

    feature_variances = torch.var(stacked_set, dim=0)
    
    return feature_variances



class VarianceWithThresholdMasksSelector():
    def __init__(self, data, mask_size, threshold=0.2):
        
        self.mask_size = mask_size
        
        trainset, _ = data
        stacked_set = torch.stack([x for x, _ in trainset])
        
        feature_variances = torch.var(stacked_set, dim=0)
        self.feat_above_thresh = (feature_variances > threshold).nonzero()[:, 0]
        
    
    def _random_choice_mask(self, mask_size):
        tmp = np.random.choice(self.feat_above_thresh, mask_size, replace=False)
        tmp.sort()

        return tmp
        
    def __call__(self, model, count):
        print(model)
        print(self)
        
        tmp = np.array([model.template.copy() for _ in range(count)])
        
        for i in range(count):
            mask = self._random_choice_mask(model.mask_size)
            tmp[i, mask] = 1
            
        masks = torch.from_numpy(tmp).to(torch.float32).to(model.device)

        return masks
        
        
        
class VarianceWithSoftmaxMasksSelector():
    def __init__(self, feature_variances, mask_size, temp_scheduler):
        
        self.mask_size = mask_size
        
        self.feature_variances = feature_variances

        self.temp_scheduler = temp_scheduler


    def _random_choice_mask(self, mask_size):
        mask = []
        unused = np.full(self.feature_variances.shape, True)
        
        for i in range(mask_size):
            
            current_temp = self.temp_scheduler.update_temp(i)
            
            probs = torch.nn.functional.softmax(self.feature_variances[unused] / current_temp, dtype=torch.float32, dim=0).numpy()

            idx = np.random.choice(unused.nonzero()[0], 1, replace=False, p=probs)
            mask.append(idx[0])
            unused[idx[0]] = False
            
        mask = np.array(mask)
        mask.sort()
    
        return mask


    def __call__(self, model, count):
        print('Mask model', model)
        
        tmp = np.array([model.template.copy() for _ in range(count)])
        
        for i in range(count):
            self.temp_scheduler.reset_to_initial()
            
            mask = self._random_choice_mask(model.mask_size)
            tmp[i, mask] = 1
            
        masks = torch.from_numpy(tmp).to(torch.float32).to(model.device)
        print('masks shape', masks.shape)
        return masks

    
                
class IncByOneTempScheduler():
    def __init__(self, initial_temp, max_temp=20, n=20):
        
        self.initial_temp = initial_temp
        self.available_temps = np.linspace(initial_temp, max_temp, n)
        
    def reset_to_initial(self, initial=None):
        if not initial:
            initial = self.initial_temp
            
        self.temp = initial
    
    def update_temp(self, n):
        self.temp = self.available_temps[n]
        
        return self.temp
        
        
class HalfHalfTempScheduler():
    def __init__(self, interval1=(-11, -1), interval2=(1, 11), n=20):
        half = n//2
        self.initial_temp = initial_temp
        self.available_temps = np.concatenate((np.linspace(*interval1, half), np.linspace(*interval2, n-half)))
        
    def reset_to_initial(self, initial=None):
        pass
    
    def update_temp(self, n):
        self.temp = self.available_temps[n]
        
        return self.temp
        
 
        
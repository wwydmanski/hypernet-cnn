
import torch
import numpy as np
from .modules import InsertableNet
import enum
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

class TrainingModes(enum.Enum):
    SLOW_STEP = "slow-step"
    CARTHESIAN = "carth"

class Hypernetwork(torch.nn.Module):
    def __init__(
        self,
        architecture=torch.nn.Sequential(torch.nn.Linear(784, 128), 
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 64)
                       ),
        target_architecture=[(20, 10), (10, 10)],
        test_nodes=100,
        mode=TrainingModes.SLOW_STEP,
        device="cuda:0",
    ):
        """ Initialize a hypernetwork.
        Args:
            target_inp_size - size of input
            out_size - size of output
            layers - list of hidden layer sizes
            test_nodes - number of test nodes
            device - device to use
        """
        super().__init__()
        self.target_outsize = target_architecture[-1][-1]
        self.mask_size = target_architecture[0][0]
        self.target_architecture = target_architecture
        self.device = device
        self.mode = mode

        self.out_size = self.calculate_outdim(target_architecture)

        self.model = architecture.to('cpu')
        gen = self.model.parameters()
        self.input_size = next(gen).size()[1]
        out_dim = self.model(torch.rand(1, self.input_size)).shape
        print(out_dim)
        output_layer = torch.nn.Linear(out_dim[1], self.out_size)
        self.model.add_module("output_layer", output_layer)
        self.model = self.model.to(device)
        
        self.dropout = torch.nn.Dropout()

        self.relu = torch.relu
        self.template = np.zeros(self.input_size)
        self.test_nodes = test_nodes
        self.test_mask = self._create_mask(test_nodes)

        self._retrained = True
        self._test_nets = None
        
    def calculate_outdim(self, architecture):
        weights = 0
        for layer in architecture:
            weights += layer[0]*layer[1]+layer[1]
        return weights

    def to(self, device):
        super().to(device)
        self.device = device
        self.test_mask = self._create_mask(self.test_nodes)
        self.model = self.model.to(device)
        return self

    def _slow_step_training(self, data, mask):
        weights = self.craft_network(mask[:1])
        mask = mask[0].to(torch.bool)
        nn = InsertableNet(
            weights[0],
            self.target_architecture,
        )

        masked_data = data[:, mask]
        res = nn(masked_data)
        return res

    def _external_mask_training(self, data, mask):
        recalculate = [True] * len(mask)
        for i in range(1, len(mask)):
            if torch.equal(mask[i - 1], mask[i]):
                recalculate[i] = False

        weights = self.craft_network(mask)
        mask = mask.to(torch.bool)

        res = torch.zeros((len(data), self.target_outsize)).to(self.device)
        for i in range(len(data)):
            if recalculate[i]:
                nn = InsertableNet(
                    weights[i],
                    self.target_architecture,
                )
            masked_data = data[i, mask[i]]
            res[i] = nn(masked_data)
        return res

    def forward(self, data, mask=None):
        """Get a hypernet prediction.
        During training we use a single target network per sample.
        During eval, we create a network for each test mask and average their results

        Args:
            data - prediction input
            mask - either None or a torch.tensor((data.shape[0], data.shape[1])).
        """
        if self.training:
            self._retrained = True
            if self.mode == TrainingModes.SLOW_STEP or self.mode == TrainingModes.CARTHESIAN:
                return self._slow_step_training(data, mask)

            if mask is None:
                mask = self._create_mask(len(data))

            return self._external_mask_training(data, mask)
        else:
            return self._ensemble_inference(data, mask)

    def _ensemble_inference(self, data, mask):
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
        res /= len(mask)
        if res.shape[1] > 1:
            res = F.softmax(res, 1)
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
            nn = InsertableNet(
                weights[i],
                self.target_architecture,
            )
            nets.append(nn)
        return nets

    @staticmethod
    def random_choice_noreplace2(l, n_sample, num_draw):
        '''
        l: 1-D array or list
        n_sample: sample size for each draw
        num_draw: number of draws

        Intuition: Randomly generate numbers, get the index of the smallest n_sample number for each row.
        '''
        l = np.array(l)
        return l[np.argpartition(np.random.rand(num_draw,len(l)), n_sample-1,axis=-1)[:,:n_sample]]
    
    def _create_mask(self, count):
        # masks = np.random.choice((len(self.template)), (count, self.mask_size), False)
        masks = Hypernetwork.random_choice_noreplace2(np.arange(len(self.template)), self.mask_size, count)
        tmp = np.array([self.template.copy() for _ in range(count)])
        for i, mask in enumerate(masks):
            tmp[i, mask] = 1
        mask = torch.from_numpy(tmp).to(torch.float32).to(self.device)
        return mask

    def craft_network(self, mask):
        out = self.model(mask)
        # out = self.output_layer(out)
        return out
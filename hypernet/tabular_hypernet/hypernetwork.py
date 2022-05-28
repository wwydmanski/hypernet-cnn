
import torch
import numpy as np
from .modules import InsertableNet
from .training_utils import get_dataloader, train_model

torch.set_default_dtype(torch.float32)


class Hypernetwork(torch.nn.Module):
    def __init__(
        self,
        inp_size=784,
        out_size=10,
        mask_size=20,
        node_hidden_size=20,
        layers=[64, 256, 128],
        test_nodes=100,
        mode="slow_step",
        device="cuda:0",
    ):
        """ Initialize a hypernetwork.
        Args:
            inp_size - size of input
            out_size - size of output
            mask_size - size of mask
            node_hidden_size - size of hidden layer in target network
            layers - list of hidden layer sizes
            test_nodes - number of test nodes
            device - device to use
        """
        super().__init__()
        self.target_outsize = out_size
        self.device = device
        self.mode = mode

        self.mask_size = mask_size
        self.input_size = inp_size
        self.node_hidden_size = node_hidden_size

        input_w_size = mask_size * node_hidden_size
        input_b_size = node_hidden_size

        hidden_w_size = node_hidden_size * out_size
        hidden_b_size = out_size

        self.out_size = input_w_size + input_b_size + hidden_w_size + hidden_b_size

        self.input = torch.nn.Linear(inp_size, layers[0])
        self.hidden1 = torch.nn.Linear(layers[0], layers[1])
        self.hidden2 = torch.nn.Linear(layers[1], layers[2])
        self.out = torch.nn.Linear(layers[2], self.out_size)

        self.dropout = torch.nn.Dropout()

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

    def _slow_step_training(self, data, mask):
        weights = self.craft_network(mask[:1])
        mask = mask[0].to(torch.bool)
        nn = InsertableNet(
            weights[0],
            self.mask_size,
            self.target_outsize,
            layers=[self.node_hidden_size],
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
                    self.mask_size,
                    self.target_outsize,
                    layers=[self.node_hidden_size],
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
            if self.mode == "slow_step":
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
                self.mask_size,
                self.target_outsize,
                layers=[self.node_hidden_size],
            )
            nets.append(nn)
        return nets

    def _create_mask(self, count):
        masks = np.array(
            [
                np.random.choice((len(self.template)), self.mask_size, False)
                for _ in range(count)
            ]
        )
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
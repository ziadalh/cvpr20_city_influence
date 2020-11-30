import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    A neural model to forecast a trend
    """
    def __init__(self, nunits_list,
                 activation_fnc=None, output_fnc=None,
                 bias=True,
                 ):
        super(MLP, self).__init__()
        if len(nunits_list) < 2:
            raise ValueError('nunits_list length must be >= 2')

        self.nlayers = len(nunits_list) - 1
        self.nunits_list = nunits_list
        self.model = None
        self.activation_fnc = activation_fnc
        self.output_fnc = output_fnc
        self.bias = bias
        self._setup()

    def _setup(self):
        nunits = self.nunits_list
        linear_list = []
        # input layer
        linear_list.append(nn.Linear(nunits[0], nunits[1], bias=self.bias))
        # hidden layers
        for i in range(1, len(nunits) - 1):
            if self.activation_fnc is not None:
                linear_list.append(self.activation_fnc())

            linear_list.append(nn.Linear(nunits[i], nunits[i+1],
                                         bias=self.bias))

        # output activations
        if self.output_fnc is not None:
            linear_list.append(self.output_fnc())

        self.model = nn.Sequential(*linear_list)

    def forward(self, input):
        return self.model(input)


class MLPGroup(nn.Module):
    """
    A neural model to forecast a group of trends
    """
    def __init__(self, list_of_nunits_list, list_of_input_splits,
                 activation_fnc=None, output_fnc=None,
                 bias=True,
                 ):
        super(MLPGroup, self).__init__()
        self.list_of_nunits_list = list_of_nunits_list
        self.list_of_input_splits = list_of_input_splits
        self.activation_fnc = activation_fnc
        self.output_fnc = output_fnc
        self.bias = bias
        self.mlps = None
        self.index_split = []
        for si in range(len(list_of_input_splits)):
            self.index_split.append(torch.as_tensor(
                                list_of_input_splits[si], dtype=torch.long))

        self._setup()

    def _setup(self):
        mlp_group = []
        lnul = self.list_of_nunits_list
        for gi in range(len(lnul)):
            mlp_group.append(MLP(lnul[gi],
                                 activation_fnc=self.activation_fnc,
                                 output_fnc=self.output_fnc,
                                 bias=self.bias,
                                 )
                             )

        self.mlps = nn.ModuleList(mlp_group)

    def forward(self, input):
        if input.dim() == 1:
            input = input.unsqueeze(0)

        mlps_output = []
        for mi in range(len(self.mlps)):
            mi_idxs = torch.as_tensor(self.index_split[mi],
                                      device=input.device)
            mi_inp = torch.index_select(input, dim=1, index=mi_idxs)
            mi_out = self.mlps[mi](mi_inp)
            if mi_out.dim() == 1:
                mi_out = mi_out.unsqueeze(0)

            mlps_output.append(mi_out)

        return torch.cat(mlps_output, dim=1)

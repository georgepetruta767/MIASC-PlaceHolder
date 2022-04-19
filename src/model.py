import torch.nn as nn

INPUT_DIM = 2
OUTPUT_DIM = 1

class TemperatureModel(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(TemperatureModel, self).__init__()
        self.hidden_layers = []

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Hidden layers
        self.hidden_layers.append(nn.Linear(INPUT_DIM, hidden_dim))
        for layer_num in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.out = nn.Linear(hidden_dim, OUTPUT_DIM)


    def forward(self, x):
        res = self.hidden_layers[0](x)

        for layer_num in range(self.num_layers - 1):
            res = self.hidden_layers[layer_num + 1](res)
            res = nn.Sigmoid(res)

        return self.out(res)

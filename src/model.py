import torch
import torch.nn as nn

INPUT_DIM = 12
NUM_CHANNELS = 3
OUTPUT_DIM = 1

class TemperatureModel(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(TemperatureModel, self).__init__()
        self.hidden_layers = []

        # "Input" layers
        self.month_layer = nn.Linear(INPUT_DIM, OUTPUT_DIM)
        self.country_layer = nn.Linear(6, OUTPUT_DIM)
        self.year_layer = nn.Linear(1, OUTPUT_DIM)

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Hidden layers
        self.hidden_layers.append(nn.Linear(NUM_CHANNELS, hidden_dim))
        for layer_num in range(self.num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.out = nn.Linear(hidden_dim, OUTPUT_DIM)

        # Activation
        self.activation = nn.Sigmoid()


    def forward(self, x):
        country = self.country_layer(x[:, 0, :6])
        month = self.month_layer(x[:, 1, :])
        year = self.year_layer(x[:, 2, :1])
        input = self.activation(torch.cat([country, month, year], dim=1))
        res = self.activation(self.hidden_layers[0](input))

        for layer_num in range(self.num_layers - 1):
            res = self.hidden_layers[layer_num + 1](res)
            res = self.activation(res)

        return self.out(res)

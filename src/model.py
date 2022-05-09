import torch.nn as nn

INPUT_DIM = 1
OUTPUT_DIM = 1

D1 = 80
D2 = 70
D3 = 60
D4 = 40
D5 = 40
D6 = 20


class TemperatureModel(nn.Module):
    def __init__(self):
        super(TemperatureModel, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(INPUT_DIM, D1),
            nn.ReLU(),
            nn.Linear(D1, D2),
            nn.ReLU(),
            nn.Linear(D2, D3),
            nn.ReLU(),
            nn.Linear(D3, D4),
            nn.ReLU(),
            nn.Linear(D4, D5),
            nn.ReLU(),
            nn.Linear(D5, D6),
            nn.ReLU(),
            nn.Linear(D6, OUTPUT_DIM),
            nn.ReLU(),
        )
        # # Hidden layers
        # self.hidden = nn.Sequential()
        # self.hidden.append(nn.Linear(INPUT_DIM, hidden_dim))
        # self.hidden.append(nn.ReLU())
        # # self.hidden_layers.append(nn.Linear(INPUT_DIM, hidden_dim))
        # for layer_num in range(num_layers - 1):
        #     # self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        #     self.hidden.append(nn.Linear(hidden_dim - decay * layer_num, hidden_dim - decay * (layer_num + 1)))
        #     self.hidden.append(nn.ReLU())
        #
        # # Output layer
        # self.hidden.append(nn.Linear(hidden_dim - decay * (num_layers - 1), OUTPUT_DIM))
        # self.hidden.append(nn.ReLU())

    def forward(self, x):
        return self.hidden(x)

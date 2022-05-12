import torch.nn as nn

INPUT_DIM = 3
OUTPUT_DIM = 1

# D1 = 80
# D2 = 70
# D3 = 60
# D4 = 40
# D5 = 40
# D6 = 20

D1 = 100
D2 = 80
D3 = 80
D4 = 60
D5 = 60
D6 = 40
D7 = 40
D8 = 20


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
            nn.Linear(D6, D7),
            nn.ReLU(),
            nn.Linear(D7, D8),
            nn.ReLU(),
            nn.Linear(D8, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.hidden(x)

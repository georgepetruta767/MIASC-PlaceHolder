import numpy
import torch
import matplotlib.pyplot as plt

from src.model import TemperatureModel
from src.preprocessing import COUNTRIES
from src.util import read_training_data, denormalize_temp, my_funct

BATCH_SIZE = 32


def load_models(epochs):
    models = []
    for epoch in epochs:
        model = TemperatureModel()
        model.load_state_dict(torch.load(f'../saved_models/epoch{epoch}_batchSize{BATCH_SIZE}.pt'))
        model.eval()

        models.append(model)

    return models


if __name__ == '__main__':
    # epochs = [0, 25, 50, 75, 100, 200, 300, 500, 700, 900]
    # epochs = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    epochs = [199]
    legend = ['actual data'] + [f'{epoch} epochs' for epoch in epochs]
    models = load_models(epochs)

    # inputs, expected_outputs, min_temp, max_temp = read_training_data()
    x_train = numpy.random.uniform(low=-10, high=10, size=1000)
    x_train.sort()
    y_train = my_funct(x_train)

    inputs = torch.from_numpy(x_train.reshape(-1, 1)).float()
    expected_outputs = torch.from_numpy(y_train.reshape(-1, 1)).float()

    # years = torch.linspace(1961, 2019, inputs.size()[0])
    # years = range(1961, 2020)
    plt.plot(x_train, expected_outputs)

    for model in models:
        output = model(inputs).detach()
        # output = denormalize_temp(output, min_temp, max_temp)
        plt.plot(x_train, output)

    plt.legend(legend)
    plt.show()


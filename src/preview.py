import torch
import matplotlib.pyplot as plt

from src.model import TemperatureModel
from src.util import read_training_data

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
    epochs = [86]
    legend = ['actual data'] + [f'{epoch} epochs' for epoch in epochs]
    models = load_models(epochs)

    inputs, expected_outputs = read_training_data()
    filtered_in, filtered_out = [], []

    for index in range(inputs.size()[0]):
        entry = inputs[index]
        if entry[0] == 1970:
            filtered_in.append(entry)
            filtered_out.append(expected_outputs[index])

    filtered_in = torch.stack(filtered_in)
    filtered_out = torch.stack(filtered_out)

    x_axis = torch.linspace(1, 12, filtered_in.size()[0])

    plt.plot(x_axis, filtered_out)

    for model in models:
        output = model(filtered_in).detach()
        # output = denormalize_temp(output, min_temp, max_temp)
        plt.plot(x_axis, output)

    plt.legend(legend)
    plt.show()


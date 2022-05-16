import torch
import matplotlib.pyplot as plt

from src.model import TemperatureModel
from src.util import read_training_data

BATCH_SIZE = 128


def load_models(epochs):
    models = []
    for epoch in epochs:
        model = TemperatureModel()
        model.load_state_dict(torch.load(f'../best_models/epoch{epoch}_batchSize{BATCH_SIZE}_8layers_noOutputReLU_3channels.pt'))
        model.eval()

        models.append(model)

    return models


if __name__ == '__main__':
    # epochs = [0, 25, 50, 75, 100, 200, 300, 500, 700, 900]
    # epochs = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    epochs = [2997]
    legend = ['actual med', 'actual max', 'actual min', 'predicted med', 'predicted max', 'predicted min'] #+ [f'{epoch} epochs' for epoch in epochs]
    models = load_models(epochs)

    inputs, expected_outputs = read_training_data()
    filtered_in, filtered_out = [], []

    for index in range(inputs.size()[0]):
        entry = inputs[index]
        if entry[0] == 1970 - 1961:
            filtered_in.append(entry)
            filtered_out.append(expected_outputs[index])

    filtered_in = torch.stack(filtered_in)
    filtered_out = torch.stack(filtered_out)

    x_axis = torch.linspace(1, 12, filtered_in.size()[0])

    plt.plot(x_axis, filtered_out[:, 0])
    plt.plot(x_axis, filtered_out[:, 1])
    plt.plot(x_axis, filtered_out[:, 2])

    for model in models:
        output = model(filtered_in).detach()
        plt.plot(x_axis, output[:, 0])
        plt.plot(x_axis, output[:, 1])
        plt.plot(x_axis, output[:, 2])

    plt.legend(legend)
    plt.show()


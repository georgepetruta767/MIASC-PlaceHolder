import torch
import matplotlib.pyplot as plt

from src.model import TemperatureModel
from src.preprocessing import COUNTRIES, MONTHS
from src.util import read_training_data, denormalize_temp, normalize, encode

HIDDEN_DIM = 100
NUM_LAYERS = 5
BATCH_SIZE = 100

COUNTRY = 'Romania'
MONTH = 'January'


def load_models(epochs):
    models = []
    for epoch in epochs:
        model = TemperatureModel(HIDDEN_DIM, NUM_LAYERS)
        model.load_state_dict(torch.load(f'../saved_models/epoch{epoch}_batchSize{BATCH_SIZE}_hiddenDim{HIDDEN_DIM}_numLayers{NUM_LAYERS}.pt'))
        model.eval()

        models.append(model)

    return models


if __name__ == '__main__':
    # epochs = [0, 25, 50, 75, 100, 200, 300, 500, 700, 900]
    epochs = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    legend = ['actual data'] + [f'{epoch} epochs' for epoch in epochs]
    models = load_models(epochs)

    inputs, expected_outputs, min_temp, max_temp = read_training_data()

    filtered_inputs, filtered_outputs = None, []
    country_normalized = encode(COUNTRY, COUNTRIES)
    month_normalized = encode(MONTH, MONTHS)

    for i in range(inputs.size()[0]):
        if inputs[i, COUNTRIES.index(COUNTRY)] == 1 and inputs[i, len(COUNTRIES) - 1 + MONTHS.index(MONTH)] == 1:
            filtered_outputs.append(expected_outputs[i])
            inp = inputs[i].reshape(1, -1)
            if filtered_inputs is None:
                filtered_inputs = inp
            else:
                filtered_inputs = torch.cat((filtered_inputs, inp))

    years = range(1961, 2020)
    plt.plot(years, filtered_outputs)

    for model in models:
        output = model(filtered_inputs).detach()
        output = denormalize_temp(output, min_temp, max_temp)
        plt.plot(years, output)

    plt.legend(legend)
    plt.show()


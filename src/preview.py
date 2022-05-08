import torch
import matplotlib.pyplot as plt

from src.model import TemperatureModel
from src.preprocessing import COUNTRIES, MONTHS
from src.util import read_training_data, encode, normalize_temp, denormalize_temp

HIDDEN_DIM = 64
NUM_LAYERS = 2

COUNTRY = 'Romania'
MONTH = 'January'

if __name__ == '__main__':
    model = TemperatureModel(HIDDEN_DIM, NUM_LAYERS)
    model.load_state_dict(torch.load('../saved_models/epoch199_batchSize40_hiddenDim64_numLayers5'))
    model.eval()
    inputs, expected_outputs, min_temp, max_temp = read_training_data()

    filtered_inputs, filtered_outputs = None, []
    country_index = COUNTRIES.index(COUNTRY)
    month_index = MONTHS.index(MONTH)

    for i in range(inputs.size()[0]):
        if inputs[i, 0, country_index] == 1 and inputs[i, 1, month_index] == 1:
            filtered_outputs.append(expected_outputs[i])
            inp = inputs[i].reshape(1, 3, 12)
            if filtered_inputs is None:
                filtered_inputs = inp
            else:
                filtered_inputs = torch.cat((filtered_inputs, inp))

    outputs = denormalize_temp(model(filtered_inputs).detach(), min_temp, max_temp)

    years = range(1961, 2020)

    plt.plot(years, filtered_outputs)
    plt.plot(years, outputs)
    plt.show()


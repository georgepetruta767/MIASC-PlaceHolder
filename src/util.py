import torch
import numpy as np

from src.preprocessing import COUNTRIES, MONTHS


def my_funct(x_input):
    y = 0.5 * x_input ** 2 + np.cos(x_input) - 10 * np.sin(5 * x_input) - 0.1 * x_input ** 3 + x_input + 100
    return y


def normalize_year(year):
    return (year - 1961) / (2020 - 1961)


def normalize_temp(temp, min_temp, max_temp):
    return (temp - min_temp) / (max_temp - min_temp)


def denormalize_temp(temp, min_temp, max_temp):
    return temp * (max_temp - min_temp) + min_temp


def normalize(elem, collection):
    index = collection.index(elem)
    if index == -1:
        raise Exception(f"Invalid element {elem}")
    max_index = len(collection) - 1

    return index / max_index


def encode(elem, collection):
    out = torch.zeros(len(collection) - 1, dtype=torch.float)
    index = collection.index(elem)
    if index == -1:
        raise Exception(f"Invalid element {elem}")
    elif index != len(collection) - 1:
        out[index] = 1
    return out


def read_training_data():
    training_input_data = None
    training_output_data = None
    with open('output.csv', 'r') as file:
        file.readline()  # skip first line

        while True:
            line = file.readline()
            if not line:
                break

            line = line.replace('\"', '')
            record = line.split(',')

            month_offset = MONTHS.index(record[1]) / 12

            for entry in range(2, len(record)):
                if record[entry] == "":
                    continue
                # 1961 (base year)
                # + entry (column offset)
                # - 2 (ignore first two columns: country and month)
                # - 1961 (for normalization)
                # + month_offset (values equally spaced in [0, 1) to represent months)
                year_month = entry - 2 + month_offset

                ts = torch.tensor([year_month])
                if training_input_data is None:
                    training_input_data = ts
                else:
                    training_input_data = torch.cat((training_input_data, ts))
                if training_output_data is None:
                    training_output_data = torch.tensor([float(record[entry])])
                else:
                    training_output_data = torch.cat((training_output_data, torch.tensor([float(record[entry])])))

    min_temp = torch.min(training_output_data)
    max_temp = torch.max(training_output_data)

    return training_input_data.reshape(-1, 1), training_output_data, min_temp, max_temp

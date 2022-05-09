import torch

from src.preprocessing import COUNTRIES, MONTHS


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

            country = encode(record[0], COUNTRIES)
            month = encode(record[1], MONTHS)

            for entry in range(2, len(record)):
                if record[entry] == "":
                    continue
                year = torch.tensor([normalize_year(1961 + entry - 2)])
                ts = torch.cat((country, month, year)).reshape(1, -1)
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

    return training_input_data, training_output_data, min_temp, max_temp

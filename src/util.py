import torch

from src.preprocessing import COUNTRIES, MONTHS


def normalize_year(year):
    return (year - 1960) / (2020 - 1960)


def normalize_temp(temp, min_temp, max_temp):
    return (temp - min_temp) / (max_temp - min_temp)


def denormalize_temp(temp, min_temp, max_temp):
    return temp * (max_temp - min_temp) + min_temp


def encode(elem, collection=None, final_dim=None):
    out = torch.zeros(len(collection) if final_dim is None else final_dim, dtype=torch.float)
    if collection is None:
        out[0] = elem
    else:
        index = collection.index(elem)
        if index == -1:
            raise Exception(f"Invalid element {elem}")
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

            country = encode(record[0], COUNTRIES, final_dim=12)
            month = encode(record[1], MONTHS, final_dim=12)

            for entry in range(2, len(record)):
                if record[entry] == "":
                    continue
                year = encode(normalize_year(1961 + entry - 2), final_dim=12)
                ts = torch.stack((
                    country,
                    month,
                    year
                ))
                ts = torch.reshape(ts, (-1, 3, 12))
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

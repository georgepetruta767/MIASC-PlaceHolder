from datetime import timedelta

import torch


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


def extract_date(date_str, data_type=float):
    split_date = date_str.split('/')
    return data_type(split_date[0]), data_type(split_date[1]), data_type(split_date[2])


def date_range(start, end):
    delta = end - start
    days = [start + timedelta(days=i) for i in range(delta.days + 1)]
    return days


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

            year, month, day = extract_date(record[0])

            year -= 1961

            med_temp = float(record[1])

            input_tensor = torch.tensor([year, month, day])
            output_tensor = torch.tensor([med_temp])
            if training_input_data is None:
                training_input_data = input_tensor
            else:
                training_input_data = torch.cat((training_input_data, input_tensor))
            if training_output_data is None:
                training_output_data = output_tensor
            else:
                training_output_data = torch.cat((training_output_data, output_tensor))

    return training_input_data.reshape(-1, 3), training_output_data

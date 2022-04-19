from src.model import TemperatureModel
import torch.nn as nn
import torch

from src.preprocessing import COUNTRIES, MONTHS

HIDDEN_DIM = 10
NUM_LAYERS = 3

NUM_EPOCHS = 200
BATCH_SIZE = 40
LEARNING_RATE = 5e-5

TRAINING_DATA_FILE = 'output.csv'

# [
#     [0, 0, 1, 0, 0, 0] - Ukraine
#     [0, 1, 0, ...0, 0] - Feb
#     [1984, 0, 0, 0...]
# ]
#
# [
#     5.3f
# ]


def encode(elem, collection=None, final_dim=None):
    out = torch.zeros(len(collection) if final_dim is None else final_dim, dtype=torch.int16)
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
                year = encode(1961 + entry - 2, final_dim=12)
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

    return training_input_data, training_output_data

if __name__ == "__main__":
    data = read_training_data()

    print(data)
    # model = TemperatureModel(HIDDEN_DIM, NUM_LAYERS).cuda()
    #
    #
    # error = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    #
    # batch_count = ... / BATCH_SIZE
    #
    # for epoch in range(NUM_EPOCHS):
    #     epoch_losses = []
    #     for batch_num in range(batch_count):
    #         input = inputs[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE].cuda()
    #         expected_output = expected_outputs[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE].cuda()
    #
    #         optimizer.zero_grad()
    #
    #         output = model(input)
    #
    #         loss = error(output, expected_output)
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_losses.append(loss.item())
    #         print(f'Epoch: {epoch} Batch: {batch_num}/{batch_count} Loss: {loss.item()}')
    #     checkpoint_name = f'epoch{epoch}_inSeqLen{in_seq_len}_outSeqLen{out_seq_len}_batchSize{BATCH_SIZE}_hiddenDim{hidden_dim}_numLayers{num_layers}'
    #     with open('losses.txt', 'a') as f:
    #         f.write(f'Model: {checkpoint_name} - LOSS: {torch.mean(torch.Tensor(epoch_losses))}\n')
    #     torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}')
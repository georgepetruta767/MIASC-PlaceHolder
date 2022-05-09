from src.model import TemperatureModel
import torch.nn as nn
import torch

from src.util import read_training_data, normalize_temp

HIDDEN_DIM = 1
NUM_LAYERS = 1

NUM_EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 1e-3

if __name__ == "__main__":
    inputs, expected_outputs, min_temp, max_temp = read_training_data()

    expected_outputs = normalize_temp(expected_outputs, min_temp, max_temp)

    model = TemperatureModel(HIDDEN_DIM, NUM_LAYERS)

    error = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())#, lr=LEARNING_RATE, momentum=0.8)
    model.train()

    batch_count = inputs.size()[0] // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        for batch_num in range(batch_count):
            input = inputs[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]
            expected_output = expected_outputs[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]

            optimizer.zero_grad()

            output = model(input)

            loss = error(output, expected_output)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            print(f'Epoch: {epoch} Batch: {batch_num}/{batch_count} Loss: {loss.item()}')
        checkpoint_name = f'epoch{epoch}_batchSize{BATCH_SIZE}_hiddenDim{HIDDEN_DIM}_numLayers{NUM_LAYERS}'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - LOSS: {torch.mean(torch.Tensor(epoch_losses))}\n')
        torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}.pt')

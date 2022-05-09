from src.model import TemperatureModel
import torch.nn as nn
import torch

from src.util import read_training_data, normalize_temp

NUM_EPOCHS = 200
BATCH_SIZE = 100
LEARNING_RATE = 1e-3

if __name__ == "__main__":
    inputs, expected_outputs, min_temp, max_temp = read_training_data()

    expected_outputs = normalize_temp(expected_outputs, min_temp, max_temp)

    model = TemperatureModel()

    error = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
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
        mean_loss = torch.mean(torch.Tensor(epoch_losses))

        checkpoint_name = f'epoch{epoch}_batchSize{BATCH_SIZE}'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - LOSS: {mean_loss}\n')
        print(f'Epoch: {epoch} Loss: {mean_loss.item()}')

        torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}.pt')

from src.model import TemperatureModel
import torch.nn as nn
import torch
import numpy

from src.util import read_training_data

NUM_EPOCHS = 3000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

if __name__ == "__main__":
    inputs, expected_outputs = read_training_data()
    inputs = inputs.cuda()
    expected_outputs = expected_outputs.cuda()

    model = TemperatureModel().cuda()

    error = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    batch_count = inputs.size()[0] // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        perm = numpy.random.permutation(inputs.size()[0])
        shuffled_input = inputs[perm]
        shuffled_output = expected_outputs[perm]
        epoch_losses = []
        for batch_num in range(batch_count):
            input = shuffled_input[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]
            expected_output = shuffled_output[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]

            optimizer.zero_grad()

            output = model(input)

            loss = error(output, expected_output)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
        mean_loss = torch.mean(torch.Tensor(epoch_losses))

        checkpoint_name = f'epoch{epoch}_batchSize{BATCH_SIZE}_8layers_noOutputReLU_3channels'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - LOSS: {mean_loss}\n')
        print(f'Epoch: {epoch} Loss: {mean_loss.item()}')

        torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}.pt')

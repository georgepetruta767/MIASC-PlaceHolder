from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from src.dataset import TemperatureDataset
from src.model import TemperatureModel
import torch.nn as nn
import torch
import numpy


NUM_EPOCHS = 3000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = .2

if __name__ == "__main__":
    dataset = TemperatureDataset()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(numpy.floor(VALIDATION_SPLIT * dataset_size))
    numpy.random.shuffle(indices)

    train_indices, validation_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

    model = TemperatureModel().cuda()

    error = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    for epoch in range(NUM_EPOCHS):
        training_losses, validation_losses = [], []
        for batch_num, (input, expected_output) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(input)

            loss = error(output, expected_output)
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())

        for batch_num, (input, expected_output) in enumerate(validation_loader):
            output = model(input)
            loss = error(output, expected_output)

            validation_losses.append(loss.item())

        training_loss = torch.mean(torch.Tensor(training_losses))
        validation_loss = torch.mean(torch.Tensor(validation_losses))

        checkpoint_name = f'epoch{epoch}_withMetrics'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - TRAINING_LOSS: {training_loss} - VALIDATION_LOSS: {validation_loss}\n')
        print(f'Epoch: {epoch} Training Loss: {training_loss.item()} Validation Loss {validation_loss.item()}')

        torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}.pt')

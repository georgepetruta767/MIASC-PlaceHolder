import numpy
from torch.utils.data import Dataset

from src.util import read_training_data


class TemperatureDataset(Dataset):
    def __init__(self):
        self.inputs, self.expected_outputs = read_training_data()
        self.inputs = self.inputs.cuda()
        self.expected_outputs = self.expected_outputs.cuda()

        perm = numpy.random.permutation(self.inputs.size()[0])
        self.inputs = self.inputs[perm]
        self.expected_outputs = self.expected_outputs[perm]

    def __getitem__(self, index):
        return self.inputs[index], self.expected_outputs[index]

    def __len__(self):
        return len(self.inputs)
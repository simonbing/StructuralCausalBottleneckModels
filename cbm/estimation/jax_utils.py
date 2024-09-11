import numpy as np
from torch.utils.data import Dataset


class CBMDataset(Dataset):
    def __init__(self, *data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        if len(self.data) == 1: # If there is only one element in the list
            return self.data[0][idx]
        else:
            return [var[idx] for var in self.data]


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

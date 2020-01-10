
from abc import ABC, abstractmethod

import torch
from torch_geometric.data import InMemoryDataset

from .base_dataset import BaseDataset

class BasePointCloud(ABC):

    def __init__(self):
        self._pos = None

    @property
    def pos(self) -> torch.tensor:
        return self._pos

    def __len__(self):
        return self.pos.shape[0]
    

class BasePointCloudDataset(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class BasePatchDataset(ABC):

    def __init__(self):
        pass

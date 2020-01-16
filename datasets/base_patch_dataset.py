
from abc import ABC, abstractmethod

import open3d as o3d
import torch
from torch_geometric.data import InMemoryDataset

from .base_dataset import BaseDataset
from utils.pointcloud_utils import build_kdtree

class BasePointCloud(ABC):

    def __init__(self, pos):
        self._pos = pos

    @property
    @abstractmethod
    def pos(self) -> torch.tensor:
        return self._pos

    def __len__(self):
        return self.pos.shape[0]
    

class BasePointCloudDataset(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> BasePointCloud:
        pass

class BasePatchDataset(ABC):

    def __init__(self):
        self._pointcloud_dataset = None

    @property
    @abstractmethod
    def pointcloud_dataset(self):
        return self._pointcloud_dataset

class BaseBallPointCloud(BasePointCloud, ABC):

    def __init__(self, pos):
        super().__init__(pos)

        self.kdtree = build_kdtree(self)

    def radius_query(self, point: torch.tensor, radius):
        k, indices, dist2 = self.kdtree.search_radius_vector_3d(point, radius)
        return k, indices, dist2

    def knn_query(self, point: torch.tensor, k):
        k, indices, dist2 = self.kdtree.search_knn_vector_3d(point, k)
        return k, indices, dist2


class BasePatchPointBallDataset(BasePatchDataset, ABC):
    '''
        Base class for patch datasets which return balls of points centered on
        points in the point clouds
    '''

    def __init__(self, pointcloud_dataset):
        super().__init__()

        self._pointcloud_dataset = pointcloud_dataset

    def __len__(self):
        return sum(len(cloud) for cloud in self.pointcloud_dataset)

    def __getitem__(self, idx):

        i = 0

        for cloud in self.pointcloud_dataset:
            cloud : BasePointCloudDataset = cloud
            if idx < i + len(cloud):
                return cloud.

    
        
    

    




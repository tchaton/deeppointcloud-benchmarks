
from abc import ABC, abstractmethod
from typing import Optional, List
import math

import open3d as o3d
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset

from base_dataset import BaseDataset
# from utils.pointcloud_utils import build_kdtree

class BasePointCloudPatchDataset(ABC, torch.utils.data.Dataset):
    '''ABC for classes which generate patches from a single pointcloud

    PointCloudPatchDatasets should be backed by a torch_geometric.data.Data object 
    with non-None pos, this is the original pointcloud which will be sampled 
    into patches. 
    '''

    def __init__(self, data : Data):
        self._data = data

        assert data.pos is not None

    @property
    def data(self) -> Data:
        return self._data

    @property
    def pos(self) -> torch.tensor:
        return self.data.pos

    def get_bounding_box(self):
        minPoint = self.pos.min(dim=0)
        maxPoint = self.pos.max(dim=0)

        return minPoint.values, maxPoint.values

class BaseMultiCloudPatchDataset(ABC, Dataset):
    '''Class representing datasets over multiple patchable pointclouds. 

    This class basically forwards methods to the underlying list of patch datasets
    '''

    def __init__(self, patchDatasets: List[BasePointCloudPatchDataset]):
        self._patchDataset = patchDatasets

    @property
    def patch_datasets(self) -> List[BasePointCloudPatchDataset]:
        return self._patchDataset

    def __len__(self):
        return sum(len(pd) for pd in self.patch_datasets)

    def __getitem__(self, idx):
        
        i = 0

        for pds in self.patch_datasets:
            if idx < i + len(pds):
                return pds[idx - i]
            i += len(pds)
    

class BasePointCloudDataset(ABC):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class BasePatchDataset(ABC):

    def __init__(self):
        self._pointcloud_dataset = None

    @property
    @abstractmethod
    def pointcloud_dataset(self):
        return self._pointcloud_dataset

class BaseBallPointCloud(BasePointCloudPatchDataset, ABC):

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

    # def __getitem__(self, idx):

    #     i = 0

    #     for cloud in self.pointcloud_dataset:
    #         cloud : BasePointCloudDataset = cloud
    #         if idx < i + len(cloud):
    #             return cloud.

# class Grid2DPatchDataset(BasePatchDataset):

#     def __init__(self, backing_dataset: Dataset):
#         super().__init__(backing_dataset)

class Grid2DPatchDataset(BasePointCloudPatchDataset):

    def __init__(self, data: Data, blockX, blockY, contextDist):
        super().__init__(data)

        self.blockX = blockX
        self.blockY = blockY
        self.contextDist = contextDist
        self.strideX = blockX - contextDist
        self.strideY = blockY - contextDist

        self.minPoint, self.maxPoint = self.get_bounding_box()

    def __len__(self):
        lenX, lenY, _ = self.maxPoint - self.minPoint
        lenX = lenX.item()
        lenY = lenY.item()

        length = math.ceil(lenX / self.strideX) * math.ceil(lenY / self.strideY)
        return length

    def get(self, idx):
        xyMin = self.minPoint
        xyMax = self.minPoint + torch.tensor([self.strideX, self.strideY, 0]).to(self.minPoint.dtype)
        index = self.get_box_index(xyMin, xyMax)

        return index

    def get_box_index(self, xyMin, xyMax):
        
        c1 = self.pos[:, 0] >= xyMin[0]
        c2 = self.pos[:, 0] <= xyMax[0]

        c3 = self.pos[:, 1] >= xyMin[1]
        c4 = self.pos[:, 1] <= xyMax[1]

        mask = c1 & c2 & c3 & c4

        return torch.arange(self.pos.shape[0])[mask]



    



    

        
    

    




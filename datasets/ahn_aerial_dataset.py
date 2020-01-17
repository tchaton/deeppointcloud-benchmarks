import os
import os.path as osp
import sys

import open3d
import torch
from torch_geometric.data import InMemoryDataset

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)

from custom_dataset.pcd_utils import AHNPointCloud
from base_patch_dataset import Grid2DPatchDataset, BaseMultiCloudPatchDataset
from base_dataset import BaseDataset

class AHNTilesDataset(InMemoryDataset):

    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):

        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            '37EN2_11.LAZ',
            '37EN2_16.LAZ',
            '37FZ2_21.LAZ',
        ] if self.split == 'train' else [
            '43FZ2_20.LAZ',
        ]

    @property
    def processed_file_names(self):
        return ['{}_data.pt'.format(self.split)]

    def download(self):
        raise NotImplementedError

    def process(self):

        data_list = []

        for raw_path in self.raw_paths:
            pcd = AHNPointCloud.from_cloud(raw_path)

            data_list.append(pcd.to_torch_data())

        torch.save(self.collate(data_list), self.processed_paths[0])

class AHNPatchDataset(Grid2DPatchDataset):

    def __init__(self, data):
        super().__init__(data, 10, 10, 0.3)

class AHNMultiCloudPatchDataset(BaseMultiCloudPatchDataset):

    def __init__(self, backingDataset):
        super().__init__([
            AHNPatchDataset(data) for data in backingDataset
        ])

class AHNAerialDataset(BaseDataset):

    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)
        self._data_path = osp.join(dataset_opt.dataroot, "AHNTilesDataset")

        self.train_dataset = AHNMultiCloudPatchDataset(
            AHNTilesDataset(self._data_path, "train"),
        )

        self.test_dataset = AHNMultiCloudPatchDataset(
            AHNTilesDataset(self._data_path, "test"),
        )





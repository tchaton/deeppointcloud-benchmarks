
import hydra

import os.path as osp
import sys
ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
sys.path.append(ROOT)

from pcd_utils import visualize_pointcloud
from datasets.utils import find_dataset_using_name


@hydra.main(config_path = osp.join(ROOT, 'conf', 'config.yaml'))
def main(cfg):
    
    dataset_name = cfg.experiment.dataset
    dataset_config = cfg.data[dataset_name]

    dataset = find_dataset_using_name(dataset_name)(dataset_config, cfg.training)

    # visualize_pointcloud(dataset.train_dataset._cloud_dataset[0])

if __name__ == '__main__':
    main()
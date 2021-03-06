![Project Logo](/docs/logo.png)

This is a framework for running common deep learning models for point cloud analysis tasks against classic benchmark. It heavily relies on pytorch geometric and hydra core.

**[Documentation](https://deeppointcloud-benchmarks.readthedocs.io/en/latest/)** | **[Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/resources.html)** | **[Facebook Hydra](https://hydra.cc/)**

The framework allows lean and yet complex model to be built with minimum effort and great reproducibility.

# COMPACT API
```yaml
# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (https://arxiv.org/abs/1706.02413)
# Credit Charles R. Qi: https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg_msg_one_hot.py

pointnet2_onehot:
      type: pointnet2_dense
      conv_type: "DENSE"
      use_category: True
      down_conv:
          module_name: PointNetMSGDown
          npoint: [1024, 256, 64, 16]
          radii: [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
          nsamples: [[16, 32], [16, 32], [16, 32], [16, 32]]
          down_conv_nn:
              [
                  [[FEAT, 16, 16, 32], [FEAT, 32, 32, 64]],
                  [[32 + 64, 64, 64, 128], [32 + 64, 64, 96, 128]],
                  [[128 + 128, 128, 196, 256], [128 + 128, 128, 196, 256]],
                  [[256 + 256, 256, 256, 512], [256 + 256, 256, 384, 512]],
              ]
      up_conv:
          module_name: DenseFPModule
          up_conv_nn:
              [
                  [512 + 512 + 256 + 256, 512, 512],
                  [512 + 128 + 128, 512, 512],
                  [512 + 64 + 32, 256, 256],
                  [256 + FEAT, 128, 128],
              ]
          skip: True
      mlp_cls:
          nn: [128, 128]
          dropout: 0.5
```

## Getting started
### Requirements:
* CUDA > 10
* Python 3 + headers (python-dev)
* [Poetry](https://poetry.eustace.io/) (Optional but highly recommended)

### Setup repo
Clone the repo to your local machine then run the following command from the root of the repo
```
poetry install
```
This will install all required dependencies in a new virtual environment.

Activate it
```
poetry shell
```
You can check that the install has been successful by running
```
python -m unittest
```

## Train pointnet++ on part segmentation task for dataset shapenet
```
poetry run python train.py experiment.model_name=pointnet2_charlesssg wandb.log=False experiment.data=shapenet
```
And you should see something like that

![logging](/docs/imgs/logging.png)

# Benchmark
## S3DIS


| Model Name | Size | Speed Train / Test | Cross Entropy | OAcc | mIou | mAcc |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [```pointnet2_original```](/benchmark/s3dis_fold5/Pointnet2_original.md)|3,026,829<!-- .element: style="text-align:center;" -->|04:29 / 01:07<!-- .element: style="text-align:center;" -->|0.0512<!-- .element: style="text-align:center;" -->|85.26<!-- .element: style="text-align:center;" -->|45.58<!-- .element: style="text-align:center;" -->|73.11 <!-- .element: style="text-align:center;" -->|


## Shapenet part segmentation
The data reported below correspond to the part segmentation problem for Shapenet for all categories. We report against mean instance IoU and mean class IoU (average of the mean instance IoU per class)

| Model Name | Use Normals | Size | Speed Train / Test | Cross Entropy | OAcc | mIou |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [```pointnet2_charlesmsg```](/benchmark/shapenet/pointnet2_charlesmsg.md) | Yes<!-- .element: style="text-align:center;" --> | 1,733,946<!-- .element: style="text-align:center;" --> | 15:07 / 01:20<!-- .element: style="text-align:center;" --> | 0.089 | 82.1<!-- .element: style="text-align:center;" --> | 85.1<!-- .element: style="text-align:center;" --> |

## Contributing
Contributions are welcome! The only asks are that you stick to the styling and that you add tests as you add more features!
For styling you can use [pre-commit hooks](https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/) to help you:
```
pre-commit install
```
A sequence of checks will be run for you and you may have to add the fixed files again to the stahed files.

## Contributers
- [Thomas Chaton](https://github.com/tchaton)
- [Nicolas Chaulet](https://github.com/nicolas-chaulet)
- [Tristan Heywood](https://github.com/tristanheywood)

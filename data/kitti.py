"""KITTI dataset placeholder.

Not yet implemented — provides the same interface as ``data.nyu`` and
``data.make3d`` so training/eval scripts can target KITTI once the dataset
pipeline is added.
"""


def get_train_loader(config):
    raise NotImplementedError("KITTI train loader is not implemented yet.")


def get_test_loader(config):
    raise NotImplementedError("KITTI test loader is not implemented yet.")

"""

This is a dataaset package.

Each dataset module exposes a uniform interface:
    from data import nyu, make3d, kitti
    train_loader = nyu.get_train_loader(config)
    test_loader  = nyu.get_test_loader(config)
"""

from data import nyu, make3d, kitti

__all__ = ["nyu", "make3d", "kitti"]

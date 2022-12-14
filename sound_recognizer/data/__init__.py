"""Module containing submodules for each dataset.
Each dataset is defined as a class in that submodule.
The datasets should have a .config method that returns
any configuration information needed by the model.
Most datasets define their constants in a submodule
of the metadata module that is parallel to this one in the
hierarchy.
"""
from .base_data_module import BaseDataModule
from .esc50 import ESC50
from .fake_images import FakeImageData

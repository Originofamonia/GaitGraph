from .preparation import DatasetSimple, DatasetDetections
from .gait import (
    CasiaBPose,
)
from .fagitue_dataset import FatigueGait


def dataset_factory(name):
    if name == "casia-b":
        return CasiaBPose
    elif name == 'fatigue':
        return FatigueGait

    raise ValueError()

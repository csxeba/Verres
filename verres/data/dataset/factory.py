from collections import namedtuple
from typing import List

import verres as V
from . import abstract
from . import cocodoom
from . import inmemory


_builders = {"cocodoom": cocodoom.COCODoomDataset,
             "mnist": inmemory.MNIST,
             "fashion_mnist": inmemory.FASHION_MNIST,
             "cifar10": inmemory.CIFAR10,
             "cifar100": inmemory.CIFAR100}


def factory(config: V.Config, specs: List[dict] = None) -> List[abstract.Dataset]:

    if specs is None:
        specs = {"training": config.training.data, "inference": config.inference.data}[config.context.execution_type]

    result = []
    for spec in specs:
        if spec.name != "cocodoom":
            raise NotImplementedError(f"Dataset not implemented: {spec.name}")
        dataset = cocodoom.COCODoomDataset(config, spec=spec)
        result.append(dataset)
        if config.context.verbose > 1:
            print(f" [Verres.dataset] - Factory built: {spec.name} for training.")

    return result

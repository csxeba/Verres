from typing import List

import verres as V
from .pipeline import Pipeline
from .codec import Codec
from . import dataset


def factory(config: V.Config, specs: List[V.config.DatasetSpec] = None) -> List[Pipeline]:

    datasets = dataset.factory(config, specs)
    pipes = []

    for ds in datasets:
        pipes.append(Pipeline(config, ds, codec=Codec(config, ds.dataset_spec.transformations)))
        if config.context.verbose > 1:
            print(f" [Verres.pipeline] - Factory built: COCODoomTrainingPipeline for training")

    return pipes

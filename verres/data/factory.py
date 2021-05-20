from typing import List

import verres as V
from .pipeline import Pipeline
from . import dataset
from . import transformation


def factory(config: V.Config, specs: List[V.config.DatasetSpec] = None) -> List[Pipeline]:

    datasets = dataset.factory(config, specs)
    pipes = []

    for ds in datasets:
        transformations = transformation.factory(config, ds.descriptor, ds.dataset_spec.transformations)
        pipes.append(Pipeline(config, ds, transformations))
        if config.context.verbose > 1:
            print(f" [Verres.pipeline] - Factory built: COCODoomTrainingPipeline for training")

    return pipes

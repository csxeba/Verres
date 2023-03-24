import cv2

import verres as V
from ... import feature
from ..sample import Sample
from . abstract import Transformation


class ImageProcessor(Transformation):

    def __init__(self, config: V.Config, transformation_spec: dict = None):
        super().__init__(
            config=config,
            transformation_spec=transformation_spec,
            input_fields="image_path",
            output_features=feature.Feature(
                "image",
                stride=1,
                sparse=False,
                dtype="uint8",
                depth=3,
                shape=tuple(config.model.input_shape_hw[:2])))

    def call(self, sample: Sample):
        image = cv2.imread(sample.input.image_path)
        if image is None:
            raise RuntimeError(f"No image found @ {sample.input.image_path}")
        assert image.shape[:2] == sample.input.shape_whc[:2][::-1] == self.cfg.model.input_shape_hw[:2]
        return {self.output_fields[0]: image}

    def decode(self, tensors, label):
        return label

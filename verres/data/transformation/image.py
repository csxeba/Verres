import cv2

import verres as V
from .. import feature
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
                shape=tuple(config.model.input_shape[:2])))

    @classmethod
    def from_descriptors(cls, config: V.Config, data_descriptor, feature_descriptor):
        return cls(config)

    def call(self, image_path):
        image = cv2.imread(image_path)
        assert image.shape[:2] == self.cfg.model.input_shape[:2]
        if image is None:
            raise RuntimeError(f"No image found @ {image_path}")
        return image

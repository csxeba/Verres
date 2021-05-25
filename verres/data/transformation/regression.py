import numpy as np

import verres as V
from .. import feature
from .abstract import Transformation


class RegressionTensor(Transformation):

    def __init__(self,
                 config: V.Config,
                 transformation_params: dict,
                 num_classes: int):

        output_features = [
            feature.Feature("regression_mask",
                            stride=transformation_params["stride"],
                            sparse=True,
                            dtype="int64",
                            depth=4,
                            shape=(None,)),
            feature.Feature("box_wh",
                            stride=transformation_params["stride"],
                            sparse=True,
                            dtype="float32",
                            depth=2,
                            shape=(None,)),
            feature.Feature("refinement",
                            stride=transformation_params["stride"],
                            sparse=True,
                            dtype="float32",
                            depth=2,
                            shape=(None,))]
        output_feature = feature.MultiFeature("regression", feature_list=output_features)

        super().__init__(
            config,
            input_fields=["bboxes", "types"],
            output_features=[output_feature])

        self.stride = transformation_params["stride"]
        self.num_classes = num_classes
        self.tensor_shape = np.array([config.model.input_shape[0] // self.stride,
                                      config.model.input_shape[1] // self.stride])

    @classmethod
    def from_descriptors(cls, config: V.Config, data_descriptor, transformation_params):
        return cls(config,
                   num_classes=data_descriptor["num_classes"],
                   transformation_params=transformation_params)

    def call(self, bboxes, types):
        shape = np.array(self.tensor_shape[:2])
        _01 = [0, 1]
        _10 = [1, 0]
        result_locations = []
        result_box_whs = []
        result_refinements = []
        for class_idx, bbox in zip(types, bboxes):
            box = np.array(bbox) / self.stride
            centroid = box[:2] + box[2:] / 2
            centroid_floored = np.round(centroid).astype(int)[::-1]
            assert centroid_floored[0] <= shape[0] and centroid_floored[1] <= shape[1]

            augmented_coords = np.stack([
                centroid_floored, centroid_floored + _01, centroid_floored + _10, centroid_floored + 1
            ], axis=0)

            in_frame = np.all([augmented_coords >= 0,
                               augmented_coords < shape[None, :]],
                              axis=(0, 2))

            augmented_coords = augmented_coords[in_frame]
            augmented_locations = np.concatenate([
                augmented_coords,
                np.full((len(augmented_coords), 1), class_idx, dtype=augmented_coords.dtype)
            ], axis=1)
            augmented_refinements = centroid[::-1][None, :] - augmented_coords
            augmented_box_whs = np.stack([box[2:][::-1]]*4, axis=0)[in_frame]

            result_locations.append(augmented_locations)
            result_box_whs.append(augmented_box_whs)
            result_refinements.append(augmented_refinements)

        result = tuple(map(
            lambda a: np.concatenate(a, axis=0), [result_locations, result_box_whs, result_refinements]))

        return result

import verres as V
from .abstract import Transformation


class FilterNumObjects(Transformation):

    def __init__(self,
                 config: V.Config,
                 transformation_spec: dict):

        input_field = ["bboxes"]
        output_field = ["_validity_flag"]

        super().__init__(config=config,
                         transformation_spec=transformation_spec,
                         input_fields=input_field,
                         output_features=[],
                         output_fields=output_field)

        self.minimum_num_objects = transformation_spec["minimum_num_objects"]

    @classmethod
    def from_descriptors(cls, config: V.Config, data_descriptor, transformation_spec):
        return cls(config=config,
                   transformation_spec=transformation_spec)

    def call(self, bboxes):
        result = len(bboxes) > 0
        return result


class FilterMapNumber(Transformation):

    def __init__(self,
                 config: V.Config,
                 transformation_spec: dict):

        input_field = ["map_no"]
        output_field = ["_validity_flag"]

        super().__init__(config=config,
                         transformation_spec=transformation_spec,
                         input_fields=input_field,
                         output_features=[],
                         output_fields=output_field)

        self.filtered_maps = set(transformation_spec["filtered_maps"])
        self.mode = transformation_spec.get("mode", "include")
        if self.mode not in {"include", "exclude"}:
            raise RuntimeError("FilterMapNumber mode must be in {include, exclude}")

    @classmethod
    def from_descriptors(cls, config: V.Config, data_descriptor, transformation_spec):
        return cls(config=config,
                   transformation_spec=transformation_spec)

    def call(self, map_no):
        validity = map_no in self.filtered_maps
        if self.mode == "exclude":
            validity = not validity
        return validity

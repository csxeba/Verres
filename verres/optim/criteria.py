import tensorflow as tf

import verres as V
from . import losses


class VRSCriteria(tf.Module):

    OUTPUT_KEYS = ()

    def __init__(self, config: V.Config, spec: dict):
        super().__init__()
        self.cfg = config
        self.spec = spec
        self.loss_weights = spec.get("loss_weights", "default")

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class ObjectDetectionCriteria(VRSCriteria):

    def __init__(self, config: V.Config, spec: dict):
        super().__init__(config, spec)
        if self.loss_weights == "default":
            self.loss_weights = {"heatmap": 1.0, "refinement": 10.0, "box_wh": 1.0}

    def call(self, ground_truth, prediction):
        heatmap_loss = losses.sse(ground_truth["heatmap"], prediction["heatmap"])
        refinement_loss = losses.sparse_vector_field_sae(ground_truth["refinement"], prediction["refinement"],
                                                         locations=ground_truth["regression_mask"])
        box_wh_loss = losses.sparse_vector_field_sae(ground_truth["box_wh"], prediction["box_wh"],
                                                     locations=ground_truth["regression_mask"])
        weighted_loss = (
                heatmap_loss * self.loss_weights["heatmap"] +
                refinement_loss * self.loss_weights["refinement"] +
                box_wh_loss * self.loss_weights["box_wh"])

        return {"heatmap": heatmap_loss, "refinement": refinement_loss, "box_wh": box_wh_loss, "loss": weighted_loss}


class CTDetCriteria(VRSCriteria):

    def __init__(self, config: V.Config, spec: dict):
        super().__init__(config, spec)
        if self.loss_weights == "default":
            self.loss_weights = {"heatmap": 1.0, "refinement": 10.0, "box_wh": 1.0}
        self.alpha = self.spec.get("alpha", 2.0)
        self.beta = self.spec.get("beta", 4.0)

    def call(self, ground_truth, prediction):

        heatmap_loss = losses.focal_loss(
            y_true=ground_truth["heatmap"],
            y_pred=prediction["heatmap"],
            alpha=self.alpha,
            beta=self.beta)
        refinement_loss = losses.sparse_vector_field_mae(
            y_true=ground_truth["refinement"],
            y_pred=prediction["refinement"],
            locations=ground_truth["regression_mask"])
        box_wh_loss = losses.sparse_vector_field_mae(
            y_true=ground_truth["box_wh"],
            y_pred=prediction["box_wh"],
            locations=ground_truth["regression_mask"])
        weighted_loss = (
                heatmap_loss * self.loss_weights["heatmap"] +
                refinement_loss * self.loss_weights["refinement"] +
                box_wh_loss * self.loss_weights["box_wh"])

        return {"heatmap": heatmap_loss, "refinement": refinement_loss, "box_wh": box_wh_loss, "loss": weighted_loss}


class PanopticSegmentationCriteria(VRSCriteria):

    def call(self, ground_truth, prediction):
        raise NotImplementedError


class SemanticSegmentationCriteria(VRSCriteria):

    def call(self, ground_truth, prediction):
        raise NotImplementedError


_mapping = {"od": ObjectDetectionCriteria,
            "ctdet_od": CTDetCriteria,
            "panoptic": PanopticSegmentationCriteria,
            "semseg": SemanticSegmentationCriteria}


def factory(config: V.Config):
    spec = config.training.criteria_spec.copy()
    name = spec.pop("name").lower()
    if config.context.verbose > 1:
        print(f" [Verres.criteria] - Factory built: {name}")

    return _mapping[name](config, spec)

import json
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import verres as V
from verres.data import Sample

DEFAULT_OUTPUT_FILES = (None, "", "default", "temporary")


def build_kpi(cocoeval: COCOeval, n_images: int, seconds: float) -> Dict[str, object]:
    """KPI dict of a summarized COCOeval: overall stats plus per-class AP over all IoUs and areas."""
    stats = cocoeval.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    precision = cocoeval.eval["precision"]  # Format: (iou, recall, class, area, maxdet)
    category_names = [cocoeval.cocoGt.cats[cid]["name"] for cid in cocoeval.params.catIds]

    kpi = {
        "AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]),
        "APs": float(stats[3]), "APm": float(stats[4]), "APl": float(stats[5]),
        "AR1": float(stats[6]), "AR10": float(stats[7]), "AR100": float(stats[8]),
        "n_images": int(n_images),
        "images_per_sec": float(n_images / seconds) if seconds > 0 else 0.,
        "per_class_AP": {},
    }

    for index, name in enumerate(category_names):
        valid = precision[:, :, index, 0, -1]  # iou 0.5:0.95, all areas, all detections
        valid = valid[valid > -1]
        kpi["per_class_AP"][name] = float(valid.mean()) if valid.size else -1.

    return kpi


class EvaluationExecutor:

    def __init__(self,
                 config: V.Config,
                 model: V.architecture.VRSArchitecture,
                 pipeline: V.data.Pipeline):

        self.cfg = config
        self.model = model
        self.pipeline = pipeline
        self.batch_size = config.evaluation.batch_size
        self.forward = tf.function(lambda x: model(x, training=False), reduce_retracing=True)

    @classmethod
    def factory(cls, config: V.Config) -> List["EvaluationExecutor"]:
        model = V.architecture.VRSArchitecture.factory(config)
        pipelines = V.data.factory(config, specs=config.evaluation.data)
        return [cls(config, model, pipeline) for pipeline in pipelines]

    def execute(self, detection_output_file: str = None) -> Dict[str, object]:
        output_file = detection_output_file or self.cfg.evaluation.detection_output_file
        if output_file in DEFAULT_OUTPUT_FILES:
            artifactory = V.Artifactory.get_default(self.cfg)
            output_file = str(artifactory.detections / "OD-detections.json")

        detections, model_seconds = self._execute_detection()
        return self._execute_evaluation(detections, output_file, model_seconds)

    def _execute_detection(self):
        """Batched forward pass; decoding stays per sample because it is not TF-traceable."""

        detections = []
        num_images = len(self.pipeline)
        model_seconds = 0.
        timer = V.utils.profiling.Timer()

        iterator = iter(self.pipeline.stream(shuffle=False, batch_size=self.batch_size, collate_batch=None))

        for start in range(0, num_images, self.batch_size):
            # The stream is infinite, so the final batch must be truncated to the remaining images.
            samples: List[Sample] = next(iterator)[:num_images - start]
            images = np.stack([sample.encoded_tensors["image"] for sample in samples])

            with timer:
                model_output = self.forward(self.model.preprocess_input(images))
            model_seconds += timer.result

            detections.extend(self._decode_batch(samples, model_output))

            print("\r [Verres] - COCO eval "
                  f"progress: {min(start + self.batch_size, num_images) / num_images:>7.2%} "
                  f"Model: {min(start + self.batch_size, num_images) / max(model_seconds, 1e-9):.2f} FPS", end="")

        print()

        return detections, model_seconds

    def _decode_batch(self, samples: List[Sample], model_output: Dict[str, tf.Tensor]) -> list:
        detections = []
        for index, sample in enumerate(samples):
            detection_label = self.pipeline.codec.decode({k: v[index] for k, v in model_output.items()})
            detections.extend(V.utils.cocodoom.generate_coco_detections(
                boxes_xy_01=detection_label.object_keypoint_coords,
                types=detection_label.object_types,
                scores=detection_label.object_scores,
                image_shape_xy=sample.input.shape_whc[:2],
                image_id=sample.ID))
        return detections

    def _execute_evaluation(self,
                            detections: list,
                            output_file: str,
                            model_seconds: float) -> Dict[str, object]:

        if len(detections) == 0:
            print(" [Verres] - No detections were generated.")
            return {}

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as file:
            json.dump(detections, file)

        coco = COCO(self.pipeline.dataset.annotation_file_path)
        det_coco = coco.loadRes(output_file)

        cocoeval = COCOeval(coco, det_coco, iouType="bbox")
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

        kpi = build_kpi(cocoeval, len(self.pipeline), model_seconds)
        kpi_path = os.path.splitext(output_file)[0] + "-kpi.json"
        with open(kpi_path, "w") as file:
            json.dump(kpi, file, indent=2)
        print(f" [Verres] - KPI written to {kpi_path}")

        return kpi

import io
import json
import pickle
from typing import List
from collections import deque

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import verres as V


class EvaluationExecutor:

    RESULT_KEYS = ["map", "map.5", "map.75", "map.S", "map.M", "map.L",
                   "mar1", "mar10", "mar100", "mar.S", "mar.M", "mar.L"]

    def __init__(self,
                 config: V.Config,
                 model: V.architecture.VRSArchitecture,
                 pipeline: V.data.Pipeline):

        self.cfg = config
        self.model = model
        self.pipeline = pipeline

    @classmethod
    def factory(cls, config: V.Config) -> List["EvaluationExecutor"]:
        model = V.architecture.VRSArchitecture.factory(config)
        pipelines = V.data.factory(config, specs=config.evaluation.data)
        return [cls(config, model, pipeline) for pipeline in pipelines]

    def execute(self, detection_output_file: str = None):
        detections = self._execute_detection()
        results = self._execute_evaluation(detections, detection_output_file)
        return results

    def _execute_detection(self):

        detections = []
        data_time = deque(maxlen=10)
        model_time = deque(maxlen=10)
        postproc_time = deque(maxlen=10)
        timer = V.utils.profiling.Timer()

        N = len(self.pipeline)

        iterator = iter(self.pipeline.stream(shuffle=False, batch_size=1, collate_batch=None))

        for i in range(1, N+1):
            with timer:
                meta = next(iterator)[0]
                image = meta["image_tensor"]
            data_time.append(timer.result)

            with timer:
                network_input = self.model.preprocess_input(image[None, ...])
                model_output = self.model(network_input, training=False)
            model_time.append(timer.result)

            with timer:
                result = self.model.postprocess(model_output)
            postproc_time.append(timer.result)

            detections.extend(V.utils.cocodoom.generate_coco_detections(
                boxes=result["boxes"],
                types=result["types"],
                scores=result["scores"],
                image_shape=image.shape[:2],
                image_id=meta["image_id"]))

            print("\r [Verres] - COCO eval "
                  f"progress: {i / N:>7.2%} "
                  f"Data: {1 / np.mean(data_time):.2f} FPS - "
                  f"MTime: {1 / np.mean(model_time):.2f} FPS - "
                  f"PTime: {1 / np.mean(postproc_time):.2f} FPS", end="")

        print()

        return detections

    def _execute_evaluation(self,
                            detections: list,
                            detection_output_file: str = None):

        if len(detections) == 0:
            print(" [Verres] - No detections were generated.")
            return np.zeros(12, dtype=float)

        if detection_output_file is None:
            if self.cfg.evaluation.detection_output_file == "default":
                artifactory = V.artifactory.Artifactory.get_default(self.cfg)
                detection_output_file = str(artifactory.detections / "OD-detections.json")
            elif self.cfg.evaluation.detection_output_file in {"tmp", "temp", "temporary"}:
                detection_output_file = io.StringIO()
            else:
                detection_output_file = self.cfg.evaluation.detection_output_file

        with open(detection_output_file, "w") as file:
            json.dump(detections, file)

        coco = COCO(self.pipeline.dataset.descriptor.annotation_file_path)
        det_coco = coco.loadRes(detection_output_file)

        cocoeval = COCOeval(coco, det_coco, iouType="bbox")
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()

        return cocoeval.stats

import itertools
import os
import pathlib

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("pycocotools")

import verres as V
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from verres.testing import fake_outputs

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_baseline.yml")
CLASS_MAPPING_PATH = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
NUM_IMAGES = 100
# Measured GT ceiling on 100 val-mini images: AP = AP50 = AP75 = 0.995 (309 of 310 objects decoded).
MIN_MAP = 0.9
MIN_MAP_50 = 0.9


def _ground_truth_detections(pipeline, num_images: int):
    """Decode the encoded ground truth as if the head had predicted it perfectly."""
    detections, image_ids = [], []
    stream = pipeline.stream(shuffle=False, batch_size=1, collate_batch=None)
    for samples in itertools.islice(stream, num_images):
        sample = samples[0]
        label = pipeline.codec.decode(fake_outputs.dense_model_output(sample))
        image_ids.append(sample.ID)
        detections.extend(V.utils.cocodoom.generate_coco_detections(
            boxes_xy_01=label.object_keypoint_coords,
            types=label.object_types,
            scores=label.object_scores,
            image_shape_xy=sample.encoded_tensors["image"].shape[1::-1],
            image_id=sample.ID))
    return detections, image_ids


def test_gt_derived_map_ceiling():
    config = V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING_PATH)
    dataset_root = config.evaluation.data[0].root
    if not os.path.isdir(dataset_root):
        pytest.skip(f"COCODoom dataset not found at {dataset_root}")

    pipeline = V.data.factory(config, specs=config.evaluation.data)[0]
    detections, image_ids = _ground_truth_detections(pipeline, NUM_IMAGES)

    ground_truth = COCO(pipeline.dataset.annotation_file_path)
    cocoeval = COCOeval(ground_truth, ground_truth.loadRes(detections), iouType="bbox")
    cocoeval.params.imgIds = image_ids  # score only the images the predictions cover
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()  # fills `stats`

    accuracy, accuracy_50 = cocoeval.stats[0], cocoeval.stats[1]
    assert accuracy_50 > MIN_MAP_50, f"mAP@0.5 dropped to {accuracy_50:.4f}"
    assert accuracy > MIN_MAP, f"mAP dropped to {accuracy:.4f}"

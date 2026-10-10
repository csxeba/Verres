import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from verres.execution.evaluation import build_kpi
from verres.utils.cocodoom import generate_coco_detections, generate_mini_dataset


def _tiny_coco_json(num_images: int, annos_per_image: int = 2) -> dict:
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 0, "name": "imp"}, {"id": 1, "name": "enemy"}],
        "images": [{"id": i, "file_name": f"run0/map00/rgb/{i:06d}.png", "width": 320, "height": 200}
                   for i in range(num_images)],
        "annotations": [{"id": i * annos_per_image + j, "image_id": i, "category_id": 0,
                         "bbox": [1., 2., 3., 4.], "segmentation": []}
                        for i in range(num_images) for j in range(annos_per_image)],
    }


def test_mini_dataset_is_evenly_strided_and_deterministic(tmp_path):
    source = tmp_path / "run-val.json"
    source.write_text(json.dumps(_tiny_coco_json(num_images=10)))
    mini = tmp_path / "run-val-mini.json"

    generate_mini_dataset(str(source), str(mini), num_images=3)

    data = json.loads(mini.read_text())
    assert [meta["id"] for meta in data["images"]] == [0, 3, 6]  # strided by 10 // 3
    assert {anno["image_id"] for anno in data["annotations"]} == {0, 3, 6}
    assert data["categories"] == _tiny_coco_json(10)["categories"]
    assert len(json.loads(source.read_text())["images"]) == 10  # source untouched

    first_pass = data["images"]
    generate_mini_dataset(str(source), str(mini), num_images=3)  # already exists, no-op
    assert json.loads(mini.read_text())["images"] == first_pass


def test_mini_dataset_caps_at_requested_size(tmp_path):
    source = tmp_path / "run-train.json"
    source.write_text(json.dumps(_tiny_coco_json(num_images=100, annos_per_image=1)))
    mini = tmp_path / "run-train-mini.json"

    generate_mini_dataset(str(source), str(mini), num_images=30)

    assert len(json.loads(mini.read_text())["images"]) == 30


def test_generate_coco_detections_keeps_top_k_by_score():
    num = 150
    boxes = np.random.rand(num, 4)
    types = np.arange(num) % 2
    scores = np.linspace(0., 1., num)

    detections = generate_coco_detections(boxes, types, scores, (320, 200), image_id=7, top_k=100)

    assert len(detections) == 100
    assert detections[0]["score"] == pytest.approx(scores[-1])  # highest score first
    assert detections[0]["category_id"] == types[-1]
    assert detections[0]["bbox"][:2] == pytest.approx([boxes[-1, 0] * 320, boxes[-1, 1] * 200])
    assert all(d["image_id"] == 7 for d in detections)


def test_generate_coco_detections_keeps_all_when_under_top_k():
    boxes = np.random.rand(4, 4)
    detections = generate_coco_detections(boxes, np.zeros(4, dtype=int), np.full(4, 0.5), (320, 200), image_id=1)
    assert len(detections) == 4


def test_build_kpi_overall_stats_and_per_class_ap():
    num_classes, num_recall = 2, 101
    precision = np.full((10, num_recall, num_classes, 4, 3), -1.)
    precision[0, :, 0, 0, -1] = 0.5   # class 0, iou 0.5, area=all, maxdet=100
    precision[1, :, 0, 0, -1] = 0.7   # class 0, iou 0.55 ... averaged in, so the class AP is 0.6
    precision[0, :, 1, 0, -1] = 0.25
    precision[0, :, 1, 1, -1] = 0.9   # must be ignored: only the "all areas" slice counts
    stats = np.array([0.3, 0.5, 0.2, -1., -1., -1., 0.1, 0.2, 0.4, -1., -1., -1.])
    cocoeval = SimpleNamespace(
        stats=stats,
        eval={"precision": precision},
        params=SimpleNamespace(catIds=[0, 1]),
        cocoGt=SimpleNamespace(cats={0: {"name": "imp"}, 1: {"name": "enemy"}}),
    )

    kpi = build_kpi(cocoeval, n_images=4, seconds=2.)

    assert kpi["AP"] == pytest.approx(0.3)
    assert kpi["AP75"] == pytest.approx(0.2)
    assert kpi["AR1"] == pytest.approx(0.1)
    assert kpi["AR100"] == pytest.approx(0.4)
    assert kpi["per_class_AP"] == {"imp": pytest.approx(0.6), "enemy": pytest.approx(0.25)}
    assert kpi["n_images"] == 4
    assert kpi["images_per_sec"] == pytest.approx(2.)


def test_build_kpi_reports_missing_class_as_minus_one():
    precision = np.full((10, 101, 1, 4, 3), -1.)
    cocoeval = SimpleNamespace(
        stats=np.full(12, -1.),
        eval={"precision": precision},
        params=SimpleNamespace(catIds=[0]),
        cocoGt=SimpleNamespace(cats={0: {"name": "imp"}}),
    )

    kpi = build_kpi(cocoeval, n_images=1, seconds=0.)

    assert kpi["per_class_AP"] == {"imp": -1.}
    assert kpi["images_per_sec"] == 0.

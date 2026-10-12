"""Ground-truth ceiling of every readout: encoding GT targets and decoding them back must be lossless.

A perfect model is emulated by feeding the encoded targets back through the real decode path; any
residual error is a bug in the encoder, the decoder or the evaluation conversion, not in training.
"""
import itertools
import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("cv2")

import cv2
import tensorflow as tf

import verres as V
from verres.data.codec import Codec
from verres.data.sample import Input, Label, Sample
from verres.data.transformation import CollateBatch
from verres.optim.criteria import VRSCriteria
from verres.testing.fake_outputs import to_logits

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = str(REPO_ROOT / "config/exp_od_overfit.yml")
CLASS_MAPPING_PATH = str(REPO_ROOT / "config/class_mappings/cocodoom.yml")
STRIDE = 4

# Small (10x10 and 10x10 at the border), medium, large-at-odd-coordinates, thin and flat boxes.
BOXES_XYXY = np.array([
    [10., 10., 20., 20.],
    [40., 50., 120., 130.],
    [301., 181., 311., 191.],
    [64., 30., 257., 170.],
    [150., 95., 160., 105.],
    [7., 33., 23., 79.],
    [100., 150., 190., 158.],
])
TYPES = np.array([0, 1, 2, 3, 0, 1, 2])
DENSE = ("refinement", "box_wh", "box_corners")


def _config(input_hw=(200, 320), peak_nms=0.1):
    config = V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING_PATH)
    config.model.input_height, config.model.input_width = input_hw
    config.model.head_spec["peak_nms"] = peak_nms
    return config


def _sample(image_wh=(320, 200), segmentation_repr=None):
    width, height = image_wh
    scale = np.array([width, height, width, height], dtype=float)
    centers = np.stack([BOXES_XYXY[:, [0, 2]].mean(1), BOXES_XYXY[:, [1, 3]].mean(1)], axis=1)
    return Sample.create(
        ID=0,
        input_=Input(image_path="synthetic.png", shape_whc=(width, height, 3)),
        label=Label(
            object_centers=centers / np.array([width, height]),
            object_types=TYPES,
            object_keypoint_coords=BOXES_XYXY / scale[None, :],
            segmentation_repr=segmentation_repr,
        ))


def _peak_cells(sample, output_hw):
    """Sparse target locations, identical to verres.operation.numeric.get_object_center_locations."""
    return V.operation.numeric.peak_cell_locations(sample.label.object_centers, output_hw)


def _perfect_outputs(sample, output_features):
    """Head outputs that decode back to `sample`'s encoded targets, as a real head would emit them."""
    outputs = {}
    grid_hw = sample.encoded_tensors.get("heatmap", np.zeros((1, 1))).shape[:2]
    cells = _peak_cells(sample, grid_hw)
    for feature in output_features:
        tensor = sample.encoded_tensors[feature]
        if feature == "heatmap":
            outputs[feature] = tf.constant(to_logits(tensor))
        elif feature in DENSE:
            dense = np.zeros(tuple(grid_hw) + (tensor.shape[-1],), dtype=np.float32)
            dense[cells[:, 0], cells[:, 1]] = tensor
            outputs[feature] = tf.constant(dense)
        elif feature == "semantic_segmentation":
            outputs[feature] = tf.constant(
                np.eye(8, dtype=np.float32)[np.asarray(tensor)[..., 0]])
        else:
            outputs[feature] = tf.constant(tensor)
    return outputs


def _decode(config, transformations, sample):
    codec = Codec(config, transformations)
    sample = codec.encode_sample(sample)
    features = [ftr.name for trf in codec.transformation_list for ftr in trf.output_features]
    return codec.decode(_perfect_outputs(sample, features)), sample


def _decoded_cells(label, grid_hw):
    """The cells the decode path used: the ones the refinement readout published, else the shared helper."""
    if label.object_center_cells is not None:
        return np.asarray(label.object_center_cells)
    return V.operation.numeric.peak_cell_locations(label.object_centers, grid_hw)


def _match_by_cell(sample, label, grid_hw, image_wh):
    """Pair every GT object with the decoded object sitting on the same (y, x, class) cell."""
    index = {(int(c), int(y), int(x)): i for i, (y, x, c) in enumerate(
        np.concatenate([_decoded_cells(label, grid_hw), np.asarray(label.object_types)[:, None]], axis=1))}

    scale = np.array([*image_wh, *image_wh])
    gt_boxes = np.asarray(sample.label.object_keypoint_coords) * scale
    gt_centers = np.asarray(sample.label.object_centers) * np.array(image_wh)
    pred_centers = np.asarray(label.object_centers) * np.array(image_wh)
    pred_boxes = np.asarray(label.object_keypoint_coords) * scale

    rows = []
    for i, (cell, gb, gc) in enumerate(zip(
            np.concatenate([_peak_cells(sample, grid_hw),
                            np.asarray(sample.label.object_types)[:, None]], axis=1), gt_boxes, gt_centers)):
        match = index.get((int(cell[2]), int(cell[0]), int(cell[1])))
        if match is None:
            continue
        intersection = (max(0., min(pred_boxes[match, 2], gb[2]) - max(pred_boxes[match, 0], gb[0]))
                        * max(0., min(pred_boxes[match, 3], gb[3]) - max(pred_boxes[match, 1], gb[1])))
        union = ((pred_boxes[match, 2] - pred_boxes[match, 0]) * (pred_boxes[match, 3] - pred_boxes[match, 1])
                 + (gb[2] - gb[0]) * (gb[3] - gb[1]) - intersection)
        rows.append(dict(
            center_error=float(np.linalg.norm(pred_centers[match] - gc)),
            size_error=float(np.abs((pred_boxes[match, 2:] - pred_boxes[match, :2]) - (gb[2:] - gb[:2])).max()),
            iou=float(intersection / union) if union > 0 else 0.,
            type=int(label.object_types[match]),
            gt_type=int(TYPES[i]),
        ))
    return rows


OD_CHAINS = {
    "uniform+refinement+box_wh": [
        {"name": "uniform_heatmap", "stride": STRIDE},
        {"name": "refinement", "stride": STRIDE},
        {"name": "box_wh", "stride": STRIDE}],
    "variable+refinement+box_wh": [
        {"name": "variable_heatmap", "stride": STRIDE},
        {"name": "refinement", "stride": STRIDE},
        {"name": "box_wh", "stride": STRIDE}],
    "uniform+box_corner": [
        {"name": "uniform_heatmap", "stride": STRIDE},
        {"name": "box_corner", "stride": STRIDE}],
}


@pytest.mark.parametrize("input_hw", [(200, 320), (208, 320)], ids=["native_320x200", "model_208x320"])
@pytest.mark.parametrize("chain_name", sorted(OD_CHAINS))
def test_od_readouts_recover_the_ground_truth_exactly(chain_name, input_hw):
    """Every OD readout must round-trip to the GT; the only loss allowed is the refinement's job."""
    config = _config(input_hw)
    grid_hw = (input_hw[0] // STRIDE, input_hw[1] // STRIDE)
    label, sample = _decode(config, OD_CHAINS[chain_name], _sample())

    rows = _match_by_cell(sample, label, grid_hw, (320, 200))
    assert len(rows) == len(BOXES_XYXY), f"{len(rows)} of {len(BOXES_XYXY)} objects decoded"
    assert [row["type"] for row in rows] == [row["gt_type"] for row in rows], "class ids were permuted"
    # Measured: 0.0 px size error and IoU 1.0 for every chain, 0.0 px centre error with a refinement.
    assert max(row["size_error"] for row in rows) < 1e-3
    assert min(row["iou"] for row in rows) > 0.999
    if "refinement" in chain_name:
        assert max(row["center_error"] for row in rows) < 1e-3
    else:
        assert max(row["center_error"] for row in rows) <= np.hypot(STRIDE, STRIDE)


def test_unrefined_centre_error_is_bounded_by_one_output_cell():
    """Without the refinement readout the centre is quantised to the cell that produced the peak."""
    config = _config()
    label, sample = _decode(config, [
        {"name": "uniform_heatmap", "stride": STRIDE},
        {"name": "box_wh", "stride": STRIDE}], _sample())

    rows = _match_by_cell(sample, label, (200 // STRIDE, 320 // STRIDE), (320, 200))
    assert len(rows) == len(BOXES_XYXY)
    assert max(row["center_error"] for row in rows) <= np.hypot(STRIDE, STRIDE)


def _segmentation_sample():
    specifications = [((40, 40, 15), 0), ((160, 100, 30), 1), ((280, 60, 10), 2), ((100, 170, 20), 3)]
    boxes = np.array([[cx - r, cy - r, cx + r, cy + r] for (cx, cy, r), _ in specifications], dtype=float)
    scale = np.array([320, 200, 320, 200], dtype=float)
    centers = np.stack([boxes[:, [0, 2]].mean(1), boxes[:, [1, 3]].mean(1)], axis=1)
    return Sample.create(
        ID=0,
        input_=Input(image_path="synthetic.png", shape_whc=(320, 200, 3)),
        label=Label(
            object_centers=centers / np.array([320, 200]),
            object_types=np.array([t for _, t in specifications]),
            object_keypoint_coords=boxes / scale[None, :],
            # COCO polygon format: one object's repr is a list of polygons.
            segmentation_repr=[[[cx - r, cy - r, cx + r, cy - r, cx + r, cy + r, cx - r, cy + r]]
                               for (cx, cy, r), _ in specifications],
        ))


@pytest.mark.parametrize("stride", [1, 4])
def test_semantic_segmentation_roundtrip(stride):
    """argmax over the decoded logits must reproduce the encoded class canvas exactly."""
    config = _config()
    label, sample = _decode(config, [{"name": "semantic_seg", "stride": stride}], _segmentation_sample())
    canvas = np.asarray(sample.encoded_tensors["semantic_segmentation"])[..., 0]
    assert np.array_equal(np.asarray(label.semantic_segmentation), canvas)


def test_semantic_segmentation_survives_a_model_input_bigger_than_the_image():
    """Masks are rasterised at 320x200 but the canvas is on the 208x320 model grid: no IndexError, same content."""
    config = _config((208, 320))
    label, sample = _decode(config, [{"name": "semantic_seg", "stride": STRIDE}], _segmentation_sample())
    canvas = np.asarray(sample.encoded_tensors["semantic_segmentation"])[..., 0]
    assert canvas.shape == (208 // STRIDE, 320 // STRIDE)
    assert np.array_equal(np.asarray(label.semantic_segmentation), canvas)


def test_instance_segmentation_roundtrip():
    """Every foreground pixel must decode back onto its own cell, affiliated with its own object."""
    config = _config()
    sample = _segmentation_sample()
    codec = Codec(config, [
        {"name": "uniform_heatmap", "stride": STRIDE},
        {"name": "semantic_seg", "stride": STRIDE},
        {"name": "instance_seg", "stride": STRIDE}])
    sample = codec.encode_sample(sample)
    canvas = np.asarray(sample.encoded_tensors["semantic_segmentation"])[..., 0]
    label = codec.decode({
        "heatmap": tf.constant(to_logits(sample.encoded_tensors["heatmap"])),
        "semantic_segmentation": tf.constant(np.eye(8, dtype=np.float32)[canvas]),
        "instance_segmentation": tf.constant(sample.encoded_tensors["instance_segmentation"]),
    })

    coords = np.asarray(label.instance_pixel_coords)  # image (WH) scale
    affiliations = np.asarray(label.instance_pixel_affiliations)
    cells = np.stack([np.round(coords[:, 1] * canvas.shape[0]).astype(int),
                      np.round(coords[:, 0] * canvas.shape[1]).astype(int)], axis=1)
    recovered = np.zeros(canvas.shape, dtype=bool)
    recovered[cells[:, 0], cells[:, 1]] = True
    assert np.array_equal(recovered, canvas > 0)

    # Every object gets its own pixels, each affiliated with the centroid of the class it carries (1-based canvas).
    gt_centers = {int(t) + 1: c for t, c in zip(sample.label.object_types, sample.label.object_centers)}
    centroids = np.asarray(label.object_centers)
    assert len(np.unique(affiliations)) == len(gt_centers)
    for affiliation in np.unique(affiliations):
        classes = np.unique(canvas[cells[affiliations == affiliation, 0], cells[affiliations == affiliation, 1]])
        assert len(classes) == 1
        assert np.abs(centroids[affiliation] - gt_centers[int(classes[0])]).max() <= 1 / min(canvas.shape)


def test_targets_and_decode_pick_the_same_peak_cell():
    """100k random centers: the training targets and the decode path must always floor to one cell."""
    config = _config()
    rng = np.random.default_rng(0)
    centers = rng.random((100_000, 2))
    # 3/50 is 3.0000000000000004 in float64 but 2.9999999 in float32: the off-by-one this guards against.
    centers[:2] = np.array([[3 / 50, 3 / 50], [0.06, 0.94]])
    sample = Sample.create(
        ID=0,
        input_=Input(image_path="synthetic.png", shape_whc=(320, 200, 3)),
        label=Label(object_centers=centers, object_types=np.zeros(len(centers), dtype=int)))
    grid_hw = (200 // STRIDE, 320 // STRIDE)

    refinement = V.data.transformation.factory(config, [{"name": "refinement", "stride": STRIDE}])[0]
    decoded = refinement.decode({"refinement": tf.zeros(grid_hw + (2,), dtype=tf.float32)}, sample.label)
    targets = V.operation.numeric.get_object_center_locations([sample], STRIDE, config.model.input_shape_hw)

    expected = np.asarray(decoded.object_center_cells)
    assert np.array_equal(targets[:, 1:], expected)
    float32_cells = np.floor(np.asarray(centers, dtype=np.float32)[:, ::-1]
                             * np.array(grid_hw, dtype=np.float32)).astype(int)
    assert not np.array_equal(float32_cells, expected), "float32 would have been numerically identical"


def test_heatmap_grid_follows_the_padded_model_input():
    """A 205x320 model input pads to 208x320, so the heatmap target is 52x80 and not 51x80."""
    config = _config((205, 320))
    sample = Codec(config, [{"name": "uniform_heatmap", "stride": STRIDE}]).encode_sample(
        _sample(image_wh=(320, 205)))
    heatmap = np.asarray(sample.encoded_tensors["heatmap"])
    assert heatmap.shape[:2] == (208 // STRIDE, 320 // STRIDE)

    cells = V.operation.numeric.peak_cell_locations(sample.label.object_centers, heatmap.shape[:2])
    assert np.all(heatmap[cells[:, 0], cells[:, 1], sample.label.object_types] == 1.)


def test_losses_vanish_for_ground_truth_predictions():
    """The criteria must see zero error when fed the encoded targets; focal is floored by its clip."""
    config = _config()
    codec = Codec(config, [
        {"name": "uniform_heatmap", "stride": STRIDE},
        {"name": "refinement", "stride": STRIDE},
        {"name": "box_wh", "stride": STRIDE}])
    samples = [codec.encode_sample(_sample()) for _ in range(2)]
    batch = CollateBatch(config, codec.output_features).process(samples)

    targets = {key: np.asarray(batch[key]) for key in ("heatmap", "refinement", "box_wh", "object_center_locations")}
    grid_hw = targets["heatmap"].shape[1:3]
    locations = targets["object_center_locations"]
    predictions = {"heatmap": np.full(targets["heatmap"].shape, -20., dtype=np.float32)}
    predictions["heatmap"][targets["heatmap"] >= 1.0] = 20.
    for feature in ("refinement", "box_wh"):
        dense = np.zeros((len(samples),) + tuple(grid_hw) + (2,), dtype=np.float32)
        for i in range(len(samples)):
            rows = locations[locations[:, 0] == i][:, 1:]
            start = int((locations[:, 0] < i).sum())
            dense[i, rows[:, 0], rows[:, 1]] = targets[feature][start:start + len(rows)]
        predictions[feature] = dense

    result = VRSCriteria(config, config.training.criteria_spec).call(
        {key: tf.constant(value) for key, value in targets.items()},
        {key: tf.constant(value) for key, value in predictions.items()})

    assert float(result["refinement"]) == 0.
    assert float(result["box_wh"]) == 0.
    # The focal loss cannot reach zero: clip_by_value(sigmoid(logits), 0.01, 0.99) leaves a floor.
    floor = float((targets["heatmap"] < 1.0).sum()) * abs(np.log(0.99)) * 0.01 ** 2
    assert float(result["heatmap"]) == pytest.approx(floor, rel=0.1)


def test_gt_targets_decode_to_a_perfect_coco_map():
    """End to end on real data: the project's own COCO eval on GT-derived detections must give 1.0."""
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    config = V.Config.from_paths(CONFIG_PATH, CLASS_MAPPING_PATH)
    if not os.path.isdir(config.evaluation.data[0].root):
        pytest.skip(f"COCODoom dataset not found at {config.evaluation.data[0].root}")

    pipeline = V.data.factory(config, specs=config.evaluation.data)[0]
    detections, image_ids = [], []
    stream = pipeline.stream(shuffle=False, batch_size=1, collate_batch=None)
    for samples in itertools.islice(stream, len(pipeline)):
        sample = samples[0]
        features = [ftr.name for trf in pipeline.codec.transformation_list for ftr in trf.output_features
                    if ftr.name != "image"]
        label = pipeline.codec.decode(_perfect_outputs(sample, features))
        image_ids.append(sample.ID)
        detections.extend(V.utils.cocodoom.generate_coco_detections(
            boxes_xy_01=label.object_keypoint_coords,
            types=label.object_types,
            scores=label.object_scores,
            image_shape_xy=sample.input.shape_whc[:2],
            image_id=sample.ID))

    ground_truth = COCO(pipeline.dataset.annotation_file_path)
    cocoeval = COCOeval(ground_truth, ground_truth.loadRes(detections), iouType="bbox")
    cocoeval.params.imgIds = image_ids
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()

    # Measured: AP = AP50 = AP75 = AR100 = 1.000 on all 22 mapped train-mini objects.
    assert cocoeval.stats[0] > 0.99, f"mAP of the GT-derived detections dropped to {cocoeval.stats[0]:.4f}"
    assert cocoeval.stats[1] > 0.99
    assert cocoeval.stats[8] > 0.99

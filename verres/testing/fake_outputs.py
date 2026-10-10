from typing import Dict

import numpy as np

from verres.data import Sample


def to_logits(probabilities: np.ndarray) -> np.ndarray:
    """Invert the head sigmoid, so a ground-truth heatmap can be fed back as logits."""
    p = np.clip(probabilities.astype(np.float32), 1e-6, 1. - 1e-6)
    return np.log(p / (1. - p))


def dense_model_output(sample: Sample) -> Dict[str, np.ndarray]:
    """Dense fake head output whose `codec.decode` recovers the encoded ground truth of `sample`."""
    tensors = dict(sample.encoded_tensors)
    tensors["heatmap"] = to_logits(tensors["heatmap"])

    # Sparse (N, 2) targets -> dense (H, W, 2) maps scattered at the heatmap peak cells.
    output_hw = np.array(tensors["heatmap"].shape[:2])
    cells = np.floor(sample.label.object_centers[:, ::-1] * output_hw).astype(int)
    for name in ("refinement", "box_wh"):
        dense = np.zeros(tuple(output_hw) + (2,), dtype=np.float32)
        dense[cells[:, 0], cells[:, 1]] = tensors[name]
        tensors[name] = dense
    return tensors

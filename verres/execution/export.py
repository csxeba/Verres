"""Deployment exports for a trained VRSArchitecture: SavedModel, TFLite and checkpoint loading."""
import pathlib
import re

import tensorflow as tf

_CHECKPOINT_NAME = re.compile(r"chkp_epoch_(\d+)_map_(\d+\.\d+)\.weights\.h5$")


def _serving_concrete(model: "tf.keras.Model"):
    """Concrete function mapping one raw [0, 255] image of the config input shape to the head outputs."""
    height, width = model.cfg.model.input_shape_hw
    signature = tf.TensorSpec([1, height, width, 3], tf.float32, name="image")
    return tf.function(lambda x: model(model.preprocess_input(x), training=False)).get_concrete_function(signature)


def export_saved_model(model, path: str) -> str:
    """Export a fixed-shape serving signature; the caller feeds raw uint8-range images."""
    tf.saved_model.save(model, path, signatures={"serving_default": _serving_concrete(model)})
    return path


def export_tflite(model, path: str, quantize: bool = False) -> str:
    """Convert the serving graph to TFLite; quantize applies dynamic-range int8 weights."""
    converter = tf.lite.TFLiteConverter.from_concrete_functions([_serving_concrete(model)], model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    pathlib.Path(path).write_bytes(converter.convert())
    return path


def _best_checkpoint(checkpoints: pathlib.Path):
    """Highest-mAP best-epoch checkpoint; filenames do not sort by mAP after a resumed run."""
    scored = []
    for path in checkpoints.glob("chkp_epoch_*_map_*.weights.h5"):
        match = _CHECKPOINT_NAME.match(path.name)
        if match:
            scored.append(((float(match.group(2)), int(match.group(1))), path))
    return max(scored)[1] if scored else None


def load_weights_into(model, run_dir: str) -> str:
    """Load a run's best (else latest) weights file into a model and return the file used."""
    checkpoints = pathlib.Path(run_dir) / "checkpoints"
    weights = _best_checkpoint(checkpoints)
    if weights is None:
        weights = checkpoints / "latest.weights.h5"
        if not weights.exists():
            raise FileNotFoundError(
                f"No weights in {checkpoints}: expected chkp_epoch_*_map_*.weights.h5 "
                f"or latest.weights.h5")
    if not model.built:
        # SmallFCNN creates nothing until a traced call, and load_weights needs built layers.
        model(tf.zeros((1, *model.cfg.model.input_shape_hw, 3), dtype=tf.float32))
    model.load_weights(str(weights))
    return str(weights)

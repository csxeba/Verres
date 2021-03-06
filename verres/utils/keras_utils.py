import os
import time
from typing import Tuple, List, Union

import tensorflow as tf


def get_default_keras_callbacks(artifactory, checkpoint_template=None):
    os.makedirs(artifactory, exist_ok=True)
    if checkpoint_template is None:
        checkpoint_dir = os.path.join(artifactory, "checkpoints")
        checkpoint_template = os.path.join(checkpoint_dir, "checkpoint_{}.h5")
    return [tf.keras.callbacks.ModelCheckpoint(checkpoint_template.format("latest")),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_template.format("best"), save_best_only=True),
            tf.keras.callbacks.CSVLogger(os.path.join(artifactory, "training_log.csv")),
            tf.keras.callbacks.TensorBoard(os.path.join(artifactory, "tensorboard"), write_graph=False)]


def plot_model_svg(model: tf.keras.Model,
                   path: str):

    from IPython.display import SVG

    data = tf.keras.utils.model_to_dot(
        model, show_shapes=True, show_layer_names=True, rankdir="TB").create("dot", "svg")
    svg = SVG(data)
    with open(path, "w") as handle:
        handle.write(svg.data)

    return svg


class ApplicationCatalogue:

    def __init__(self):
        self.application_init_file_path = os.path.join(tf.keras.applications.__path__[0], "__init__.py")
        self.applications = []
        self._populate_catalogue()

    def _populate_catalogue(self):
        with open(self.application_init_file_path) as handle:
            for line in handle:
                if "from tensorflow." not in line:
                    continue
                self.applications.append(line.split(" ")[-1].strip())
        self.applications.sort()

    def make_model(self,
                   model_name: str,
                   include_top: bool = False,
                   input_shape: tuple = None,
                   fixed_batch_size: int = None,
                   build_model: bool = True,
                   weights: Union[None, str] = None) -> tf.keras.Model:

        if model_name not in self.applications:
            raise ValueError(f"{model_name} is not in the catalogue of applications.")

        if input_shape is None:
            input_shape = (None, None, 3)

        input_tensor = tf.keras.Input(input_shape, fixed_batch_size, dtype=tf.float32)

        model = getattr(tf.keras.applications, model_name)(include_top=include_top,
                                                           weights=weights,
                                                           input_shape=input_shape,
                                                           input_tensor=input_tensor)
        if build_model:
            input_tensor = tf.zeros((1,) + input_shape, dtype=tf.float32)
            model(input_tensor)  # builds the forward-pass

        return model


def measure_forward_fps(model: tf.keras.Model,
                        input_tensor: tf.Tensor,
                        repeats: int = 100,
                        verbose: int = 1):

    if verbose:
        print(f" [*] Measuring {model.__class__.__name__} FPS...")

    times = []
    for i in range(repeats+1):
        start = time.time()
        model(input_tensor)
        times.append(time.time() - start)
        if verbose:
            print(f"\rProgress: {(i + 1) / 101:>7.2%}", end="")
    if verbose:
        print()

    mean_time = tf.reduce_mean(times[1:]).numpy()
    if verbose:
        print(f" [*] FPS: {1 / mean_time:.5f}")
        print(f" [*] SPF: {mean_time:.5f}")
        print()

    return mean_time


def measure_backwards_fps(model: tf.keras.Model,
                          data: Tuple[Tuple[tf.Tensor]],
                          repeats: int = 100,
                          verbose: int = 1):

    times = []
    for i in range(repeats+1):
        start = time.time()
        model.train_step(data)
        times.append(time.time() - start)
        if verbose:
            print(f"\rProgress: {(i + 1) / 101:>7.2%}", end="")
    if verbose:
        print()

    mean_time = tf.reduce_mean(times[1:]).numpy()
    if verbose:
        print(f" [*] BATCH PER SEC: {1 / mean_time:.5f}")
        print(f" [*] SEC PER BATCH: {mean_time:.5f}")
        print()

    return mean_time


def touch_layers(model: tf.keras.Model, callback: callable):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Layer):
            callback(layer)
        if hasattr(layer, "layers"):
            touch_layers(layer, callback)


def inject_regularizer(model,
                       kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                       bias_regularizer: tf.keras.regularizers.Regularizer = None,
                       included_layers: Union[str, List[str]] = "all",
                       excluded_layers: Union[None, List[str]] = None):

    def _inject_regularizer_callback(layer: tf.keras.layers):
        if included_layers != "all" and layer.name not in included_layers:
            return
        if excluded_layers is not None and layer.name in excluded_layers:
            return
        if hasattr(layer, "kernel_regularizer"):
            layer.kernel_regularizer = kernel_regularizer
        if hasattr(layer, "bias_regularizer"):
            layer.bias_regularizer = bias_regularizer

    touch_layers(model, callback=_inject_regularizer_callback)

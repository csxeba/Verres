import os
import time
from typing import Tuple, List, Union

import tensorflow as tf


def plot_model_svg(model: tf.keras.Model,
                   path: str):

    from IPython.display import SVG

    data = tf.keras.utils.model_to_dot(
        model, show_shapes=True, show_layer_names=True, rankdir="TB").create("dot", "svg")
    svg = SVG(data)
    with open(path, "w") as handle:
        handle.write(svg.data)

    return svg


class _KerasApplicationDescriptor:

    def __init__(self,
                 application_type_str: str,
                 preprocess_fn_provider_str: str):

        self.application_type_str = application_type_str
        self.preprocess_fn_provider_str = preprocess_fn_provider_str

    def get(self,
            include_top: bool = False,
            input_shape: tuple = None,
            fixed_batch_size: int = None,
            build_model: bool = True,
            weights: Union[None, str] = None):

        if input_shape is None:
            input_shape = (None, None, 3)

        input_tensor = tf.keras.Input(input_shape, fixed_batch_size, dtype=tf.float32)

        application_type = getattr(tf.keras.applications, self.application_type_str)
        application = application_type(include_top=include_top,
                                       weights=weights,
                                       input_shape=input_shape,
                                       input_tensor=input_tensor)
        preprocess_fn_provider_module = getattr(tf.keras.applications, self.preprocess_fn_provider_str)
        application.preprocess_input = preprocess_fn_provider_module.preprocess_input
        return application


class ApplicationCatalogue:

    def __init__(self):
        self.application_init_file_path = os.path.join(tf.keras.applications.__path__[0], "__init__.py")
        self.applications = {}
        self._populate_catalogue()

    def _populate_catalogue(self):
        with open(self.application_init_file_path) as handle:
            for line in handle:
                if "from tensorflow." not in line:
                    continue
                words = line.split(" ")
                application_type_str = words[-1].strip()
                preprocess_fn_provider_str = words[1].split(".")[-1]
                self.applications[application_type_str] = _KerasApplicationDescriptor(
                    application_type_str=application_type_str,
                    preprocess_fn_provider_str=preprocess_fn_provider_str)

    def make_model(self,
                   model_name: str,
                   include_top: bool = False,
                   input_shape: tuple = None,
                   fixed_batch_size: int = None,
                   build_model: bool = True,
                   weights: Union[None, str] = None) -> tf.keras.Model:

        if model_name not in self.applications:
            err = [f"{model_name} is not in the catalogue of applications.",
                   f"Available applications: {', '.join(self.applications)}"]
            raise ValueError("\n".join(err))
        application_descripor: _KerasApplicationDescriptor = self.applications[model_name]

        model = application_descripor.get(include_top=include_top,
                                          input_shape=(None, None, 3),
                                          fixed_batch_size=fixed_batch_size,
                                          build_model=build_model,
                                          weights=weights)
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


def deduce_feature_spec_params(model: Union[tf.keras.Model, tf.keras.layers.Layer]) -> dict:

    input_tensor = tf.zeros([1, 960, 960, 3], dtype=tf.float32)
    output_tensor = model(input_tensor)
    output_shape = tf.keras.backend.int_shape(output_tensor)[1]
    working_stride = 960 // output_shape

    result = {"layer_name": model.layers[-1].name,
              "working_stride": working_stride,
              "width": output_shape}

    print(f" [Verres] - automatically deduced feature layer: {result['layer_name']}")

    return result

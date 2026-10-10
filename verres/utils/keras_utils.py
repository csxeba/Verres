import os
import re
from typing import Union

import tensorflow as tf

# Keras 3 autogenerates `from <package> import <Class> as <Class>`, wrapped in parentheses when long.
_APPLICATION_IMPORT = re.compile(r"from\s+([\w.]+)\s+import\s*\(?\s*([A-Z]\w*)\s+as\s+\2")


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
            source = handle.read()
        for module_path, application_type_str in _APPLICATION_IMPORT.findall(source):
            self.applications[application_type_str] = _KerasApplicationDescriptor(
                application_type_str=application_type_str,
                preprocess_fn_provider_str=module_path.split(".")[-1])

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

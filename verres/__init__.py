import os
import pathlib

# Keras reads the backend on import, so it has to be chosen before anything pulls keras in.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras

if int(keras.__version__.split(".")[0]) < 3:
    raise RuntimeError(
        f"Verres requires Keras 3 with the TensorFlow backend, but keras {keras.__version__} is "
        "imported. Install tensorflow >= 2.16 (which ships Keras 3) and uninstall tf_keras, which "
        "takes over tf.keras when present.")

if keras.backend.backend() != "tensorflow":
    raise RuntimeError(
        f"Verres requires the tensorflow Keras backend, but `{keras.backend.backend()}` is active. "
        "Set KERAS_BACKEND=tensorflow before the first `import keras`; in a notebook the kernel may "
        "have imported keras already, so Runtime -> Restart session.")

from .config import Config
from .artifactory import Artifactory

from . import utils
from . import operation
from . import data
from . import optim
from . import architecture
from . import visualization
from . import execution

working_mode = "library"
_current = pathlib.Path.cwd()
if "verres" in {part.lower() for part in _current.parts}:
    directory = _current
    for name in reversed(_current.parts):
        if name.lower() == "verres":
            os.chdir(directory)
            working_mode = "framework"
        directory = directory.parent

print(f" [Verres] - Running in {working_mode} mode!")
print(f" [Verres] - CWD:", os.getcwd())

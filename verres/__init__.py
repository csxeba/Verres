import importlib.metadata
import importlib.util
import os

# Verres targets the legacy Keras 2 API. TF >= 2.16 exposes it only through tf_keras, but on older
# TF the same variable makes tf.keras unimportable, so the installed TF version decides - read from
# the distribution metadata, since importing tensorflow here would be too late for the variable.
_installed_tf = (0, 0)
for _distribution in ("tensorflow", "tensorflow-cpu"):
    try:
        _installed_tf = tuple(int(part) for part in importlib.metadata.version(_distribution).split(".")[:2])
        break
    except importlib.metadata.PackageNotFoundError:
        pass

if _installed_tf >= (2, 16) and importlib.util.find_spec("tf_keras") is not None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import pathlib

import tensorflow as tf

try:
    # TF reads TF_USE_LEGACY_KERAS when `tensorflow` is imported, but binds tf.keras to Keras 3
    # anyway on the next *submodule* import - which is the form the package itself uses below.
    # Checking through that same import is what makes a kernel that imported TF too early visible.
    from tensorflow.keras import layers as _layers
    keras_package = _layers.__name__.split(".")[0]
except Exception as error:  # this tensorflow has no importable tf.keras at all
    raise RuntimeError(
        "Verres requires the legacy Keras 2 API and tf.keras failed to load. On tensorflow >= 2.16 "
        f"install the tf_keras package matching {tf.__version__}; on older tensorflow make sure "
        f"TF_USE_LEGACY_KERAS is unset. Cause: {error}") from error

# TF < 2.16 ships the legacy API in-tree; from 2.16 on it is only reachable through tf_keras.
if tuple(int(part) for part in tf.__version__.split(".")[:2]) >= (2, 16) and keras_package != "tf_keras":
    raise RuntimeError(
        f"Verres requires the legacy Keras 2 API, but tf.keras resolved to {keras_package} "
        f"(tensorflow {tf.__version__}). Install tf_keras and set TF_USE_LEGACY_KERAS=1 before "
        "the first `import tensorflow`; in a notebook TensorFlow is already imported by the "
        "kernel, so Runtime -> Restart session after installing.")

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

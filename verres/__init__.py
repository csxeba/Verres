import os
import pathlib

from .config import Config
from .artifactory import Artifactory

from . import utils
from . import operation
from . import data
from . import optim
from . import architecture
from . import methods
from . import visualization
from . import execution

try:
    import tfkerassurgeon
except ImportError:
    print(" [Verres] - tfkerassurgeon not available. Pruning is disabled.")
else:
    from . import pruning

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

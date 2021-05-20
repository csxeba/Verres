import os

from artifactorium import Artifactorium as _Artifactorium

import verres as V


class Artifactory(_Artifactorium):

    __slots__ = "checkpoints", "tensorboard", "logfile_path", "detections"

    default_instance = None

    def __init__(self, config: V.Config):

        if self.__class__.default_instance is not None:
            raise RuntimeError("Attempted to rebuild the Verres Artifactory, which is a singleton.")

        if not config.context.artifactory_root:
            raise RuntimeError("Please set context.artifactory_root in your config YAML.")
        if not config.context.experiment_set:
            raise RuntimeError("Please set context.experiment_set in your config YAML.")
        if not config.context.experiment_name:
            raise RuntimeError("Please set context.experiment_name in your config YAML.")

        args = [config.context.artifactory_root,
                config.context.experiment_set,
                config.context.experiment_name,
                "NOW"]

        super().__init__(*args, return_paths_as_string=False)

        self.register_path("checkpoints")
        self.register_path("tensorboard")
        self.register_path("detections")
        self.register_path("logfile_path", "training_logs.csv", is_file=True)

        print(f" [Verres.Artifactory] - Root set to {self.root}")

        if self.__class__.default_instance is None:
            self.__class__.default_instance = self

    def make_checkpoint_template(self, model_name=""):
        return os.path.join(self.checkpoints, "{}_chkp_{}.h5"
                                              "".format(model_name, "{}"))

    @classmethod
    def get_default(cls, config: V.Config):
        if cls.default_instance is None:
            return cls(config)
        return cls.default_instance

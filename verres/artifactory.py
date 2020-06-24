import os

from artifactorium import Artifactorium


class Artifactory(Artifactorium):

    __slots__ = "checkpoints", "tensorboard", "logfile_path"

    def __init__(self, root="default", experiment_name=None):

        if root == "default":
            current = os.path.split(os.getcwd())[-1]
            if current in ["experiments", "keepers"]:
                os.chdir("..")
            root = "artifactory"

        super().__init__(root, experiment_name, "NOW")

        self.register_path("checkpoints")
        self.register_path("tensorboard")
        self.register_path("logfile_path", "training_logs.csv", is_file=True)

        print(f"[Artifactory] - Root set to {self.root}")

    def make_checkpoint_template(self, model_name=""):
        return os.path.join(self.checkpoint_root, "{}_chkp_{}".format(model_name, "{}"))

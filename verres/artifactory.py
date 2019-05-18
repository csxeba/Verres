import os
import pathlib
import datetime


class Artifactory:

    def __init__(self, root=None):
        if root is None:
            root = self._default_root()
        self.root = pathlib.Path(root)
        self.experiment_root = self.root/datetime.datetime.now().strftime("xp_%Y%m%d.%H%M%S")
        self.checkpoint_root = self.experiment_root/"checkpoints"
        self.tensorboard_root = self.experiment_root/"tensorboard"
        for roots in [self.checkpoint_root, self.tensorboard_root]:
            roots.mkdir(parents=True, exist_ok=True)
        self.logfile_path = self.experiment_root/"training_logs.csv"

    @staticmethod
    def _default_root():
        current = os.getcwd()
        if "experiments" in current:
            os.chdir("..")
        return os.path.join(current, "artifactory")

    def make_checkpoint_template(self, model_name=""):
        return os.path.join(self.checkpoint_root, "{}_chkp_{}".format(model_name, "{}"))

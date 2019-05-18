import pathlib
import datetime


class Artifactory:

    def __init__(self):
        current = pathlib.Path.cwd()
        if os.path.join("verres", "experiments") in os.getcwd():
            os.chdir("..")
        self.root = os.path.abspath("./artifactory")
        self.experiment_root = os.path.join(self.root, datetime.datetime.now().strftime("xp_%Y%m%d.%H%M%S"))
        self.checkpoint_root = os.path.join(self.experiment_root, "checkpoints")
        self.tensorboard_root = os.path.join(self.experiment_root, "tensorboard")
        os.makedirs("artifactory")

    @property
    def tensorboard_root(self):
        return

    @property
    def logfile_path(self):
        return os.path.join(self.experiment_root, "training_logs.csv")

    def make_checkpoint_template(self, model_name=""):
        return os.path.join(self.checkpoint_root, "{}_chkp_{}".format(model_name, "{}"))

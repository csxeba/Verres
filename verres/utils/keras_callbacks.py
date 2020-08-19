import tensorflow as tf

from verres.artifactory import Artifactory
from verres.data.cocodoom import evaluation, COCODoomLoader


class ResetOptimizerState(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        self.model.optimizer = self.model.optimizer.from_config(self.model.optimizer.get_config())


class ObjectMAP(tf.keras.callbacks.Callback):

    def __init__(self, loader: COCODoomLoader, artifactory: Artifactory = None, checkpoint_best: bool = True):
        super().__init__()
        self.loader = loader
        if artifactory is None:
            artifactory = Artifactory.get_default()
        self.artifactory = artifactory
        self.detection_tmp = str((artifactory.detections / "detections_epoch_{epoch}.json"))
        self.checkpoint_tmp = str((artifactory.checkpoints / "chkp_epoch_{epoch}_map_{map:.4f}.h5"))
        self.last_map = -1.
        self.last_chkp = ""
        self.checkpoint_best = checkpoint_best

    def on_epoch_end(self, epoch, logs=None):
        result = evaluation.run(loader=self.loader, model=self.model, detection_file=self.detection_tmp.format(epoch=epoch+1))
        mAP = result[0]
        if self.checkpoint_best:
            if mAP > self.last_map:
                checkpoint_path = self.checkpoint_tmp.format(epoch=epoch+1, map=mAP)
                self.model.save_weights(checkpoint_path)
                self.last_map = mAP
                self.last_chkp = checkpoint_path
        logs = logs or {}
        logs["mAP"] = mAP
        logs["mAR"] = result[6]
        return logs

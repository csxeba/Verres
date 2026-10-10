import os
from typing import List, Union

import tensorflow as tf

import verres as V


class ObjectMAP(tf.keras.callbacks.Callback):

    def __init__(self,
                 config: V.Config,
                 checkpoint_best: bool = True,
                 every_n_epochs: int = 1):

        super().__init__()
        artifactory = V.Artifactory.get_default(config)
        self.cfg = config
        self.evaluator: Union[V.execution.EvaluationExecutor, None] = None
        self.artifactory = artifactory
        self.checkpoint_tmp = str((artifactory.checkpoints / "chkp_epoch_{epoch}_map_{map:.4f}.h5"))
        self.last_map = -1.
        self.last_chkp = ""
        self.checkpoint_best = checkpoint_best
        self.every_n_epochs = every_n_epochs

    def set_model(self, model):
        super().set_model(model)
        pipelines = V.data.factory(self.cfg, self.cfg.evaluation.data)
        self.evaluator = V.execution.EvaluationExecutor(self.cfg, model, pipelines[0])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        result = {}
        if (epoch + 1) % self.every_n_epochs == 0:
            detection_output_file = str(self.artifactory.detections / "detections.json")
            result = self.evaluator.execute(detection_output_file)

        if self.checkpoint_best and result.get("AP", -1.) > self.last_map:
            self._save_best(epoch, result["AP"])

        # The CSV log freezes its columns at the end of the first epoch, so the keys are always published.
        logs["cocodoom_val/mAP"] = result.get("AP", -1.)
        logs["cocodoom_val/mAP50"] = result.get("AP50", -1.)
        logs["cocodoom_val/mAR100"] = result.get("AR100", -1.)
        return logs

    def _save_best(self, epoch: int, mAP: float):
        """Weights-only best checkpoint; the previous best survives until the new file is complete."""
        checkpoint_path = self.checkpoint_tmp.format(epoch=epoch+1, map=mAP)
        temporary_path = checkpoint_path + ".incomplete.h5"
        self.model.save_weights(temporary_path, overwrite=True)
        os.replace(temporary_path, checkpoint_path)
        if self.last_chkp and os.path.exists(self.last_chkp):
            os.remove(self.last_chkp)
        self.last_map = mAP
        self.last_chkp = checkpoint_path


class LatestModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, config: V.Config):

        artifactory = V.Artifactory.get_default(config)
        super().__init__(filepath=str(artifactory.checkpoints / "latest.h5"),
                         monitor="loss",
                         verbose=0,
                         save_best_only=False,
                         save_weights_only=True)


class BestModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self,
                 config: V.Config,
                 monitored_kpi: str,
                 monitoring_mode: str):

        artifactory = V.Artifactory.get_default(config)
        super().__init__(
            filepath=str(artifactory.checkpoints / f"best_{monitored_kpi.replace('/', '_')}.h5"),
            monitor=monitored_kpi,
            mode=monitoring_mode,
            verbose=0,
            save_best_only=True,
            save_weights_only=True)


class CSVLogger(tf.keras.callbacks.CSVLogger):

    def __init__(self, config: V.Config, **kwargs):
        artifactory = V.Artifactory.get_default(config)
        filename = artifactory.logfile_path
        super().__init__(filename, **kwargs)


class TensorBoard(tf.keras.callbacks.TensorBoard):

    def __init__(self, config: V.Config, **kwargs):
        artifactory = V.Artifactory.get_default(config)
        profile_batch = kwargs.pop("profile_batch", 0)
        super().__init__(log_dir=artifactory.tensorboard, profile_batch=profile_batch, **kwargs)


_mapping = {
    "ObjectMAP": ObjectMAP,
    "LatestModelCheckpoint": LatestModelCheckpoint,
    "BestModelCheckpoint": BestModelCheckpoint,
    "CSVLogger": CSVLogger,
    "TensorBoard": TensorBoard}


def factory(config: V.Config) -> List[tf.keras.callbacks.Callback]:

    callbacks = []

    for callback_spec in config.training.callbacks:
        spec = callback_spec.copy()
        callback_name = spec.pop("name")
        callback_type = _mapping.get(callback_name, None)
        if callback_type is None:
            callback_type = getattr(tf.keras.callbacks, callback_name, None)
        else:
            spec["config"] = config
        if callback_type is None:
            raise NotImplementedError(f"Specified callback is not available in Verres: {callback_name}")
        callback = callback_type(**spec)
        if config.context.verbose > 1:
            print(f" [Verres.callbacks] - Factory built: {callback_name}")
        callbacks.append(callback)

    return callbacks

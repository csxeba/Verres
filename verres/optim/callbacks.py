import os
from typing import List, Union

import tensorflow as tf

import verres as V


class ResetOptimizerState(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        self.model.optimizer = self.model.optimizer.from_config(self.model.optimizer.get_config())


class ObjectMAP(tf.keras.callbacks.Callback):

    def __init__(self,
                 config: V.Config,
                 checkpoint_best: bool = True):

        super().__init__()
        artifactory = V.Artifactory.get_default(config)
        self.cfg = config
        self.evaluator: Union[V.execution.EvaluationExecutor, None] = None
        self.artifactory = artifactory
        self.detection_tmp = str((artifactory.detections / "detections_epoch_{epoch}.json"))
        self.checkpoint_tmp = str((artifactory.checkpoints / "chkp_epoch_{epoch}_map_{map:.4f}.h5"))
        self.file_writer = tf.summary.create_file_writer(str(artifactory.tensorboard))
        self.last_map = -1.
        self.last_chkp = ""
        self.checkpoint_best = checkpoint_best

    def set_model(self, model):
        super().set_model(model)
        pipelines = V.data.factory(self.cfg, self.cfg.evaluation.data)
        self.evaluator = V.execution.EvaluationExecutor(self.cfg, model, pipelines[0])

    def on_epoch_end(self, epoch, logs=None):
        detection_output_file = self.detection_tmp.format(epoch=epoch)
        result = self.evaluator.execute(detection_output_file)

        mAP = result[0]
        if self.checkpoint_best:
            if mAP > self.last_map:
                checkpoint_path = self.checkpoint_tmp.format(epoch=epoch+1, map=mAP)
                self.model.save_weights(checkpoint_path, overwrite=True)
                if os.path.exists(self.last_chkp):
                    os.remove(self.last_chkp)
                self.last_map = mAP
                self.last_chkp = checkpoint_path
        with self.file_writer.as_default():
            tf.summary.scalar("cocodoom_val/mAP", mAP, step=epoch)
            tf.summary.scalar("cocodoom_val/mAR", result[6], step=epoch)

        logs["cocodoom_val/mAP"] = mAP
        logs["cocodoom_val/mAR"] = result[6]
        return logs


class LossAggregator(tf.keras.callbacks.Callback):

    def __init__(self, names: List[str]):
        super().__init__()
        self.names = names
        self.variables = [0. for _ in names]

    def on_batch_end(self, batch, logs=None):
        for v, name in zip(self.variables, self.names):
            data = logs.pop(name)
            v += data
            logs[name] = v / (batch+1)

    def on_epoch_end(self, epoch, logs=None):
        for i, v in self.variables:
            self.variables[i] = 0.


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
            save_best_only=False,
            save_weights_only=True)


class CSVLogger(tf.keras.callbacks.CSVLogger):

    def __init__(self, config: V.Config):
        artifactory = V.Artifactory.get_default(config)
        filename = artifactory.logfile_path
        super().__init__(filename)


_mapping = {
    "ResetOptimizerState": ResetOptimizerState,
    "ObjectMAP": ObjectMAP,
    "LossAggregator": LossAggregator,
    "LatestModelCheckpoint": LatestModelCheckpoint,
    "BestModelCheckpoint": BestModelCheckpoint,
    "CSVLogger": CSVLogger}


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

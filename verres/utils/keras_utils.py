import os

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard


def get_default_keras_callbacks(artifactory, checkpoint_template=None):
    os.makedirs(artifactory, exist_ok=True)
    if checkpoint_template is None:
        checkpoint_dir = os.path.join(artifactory, "checkpoints")
        checkpoint_template = os.path.join(checkpoint_dir, "checkpoint_{}.h5")
    return [ModelCheckpoint(checkpoint_template.format("latest")),
            ModelCheckpoint(checkpoint_template.format("best"), save_best_only=True),
            CSVLogger(os.path.join(artifactory, "training_log.csv")),
            TensorBoard(os.path.join(artifactory, "tensorboard"), write_graph=False)]

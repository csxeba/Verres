import os

import tensorflow as tf


def get_default_keras_callbacks(artifactory, checkpoint_template=None):
    os.makedirs(artifactory, exist_ok=True)
    if checkpoint_template is None:
        checkpoint_dir = os.path.join(artifactory, "checkpoints")
        checkpoint_template = os.path.join(checkpoint_dir, "checkpoint_{}.h5")
    return [tf.keras.callbacks.ModelCheckpoint(checkpoint_template.format("latest")),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_template.format("best"), save_best_only=True),
            tf.keras.callbacks.CSVLogger(os.path.join(artifactory, "training_log.csv")),
            tf.keras.callbacks.TensorBoard(os.path.join(artifactory, "tensorboard"), write_graph=False)]


def to_tpu(keras_model):
    TPU_ADDR = os.environ.get("COLAB_TPU_ADDR")
    if TPU_ADDR is None:
        print("[W] No TPU available, returning normal model!")
        return keras_model
    TPU_WORKER = "grpc://" + TPU_ADDR

    tpu_model = tf.contrib.tpu.keras_to_tpu_model(
        keras_model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
        ))
    return tpu_model

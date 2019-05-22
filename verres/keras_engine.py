from enum import Enum


class ENGINE(Enum):

    KERAS = 1
    TENSORFLOW = 2


def get_engine(engine_type: ENGINE=None):
    if engine_type is None:
        engine_type = ENGINE.KERAS
    if not isinstance(engine_type, ENGINE):
        engine = engine_type
    if engine_type == ENGINE.KERAS:
        import keras
        engine = keras
    else:
        import tensorflow as tf
        engine = tf.keras
    return engine


def to_tpu(keras_model):
    import os
    import tensorflow as tf

    TPU_ADDR = os.environ.get("COLAB_TPU_ADDR")
    if TPU_ADDR is None:
        raise RuntimeError("No TPU available!")
    TPU_WORKER = "grpc://" + TPU_ADDR

    tpu_model = tf.contrib.tpu.keras_to_tpu_model(
        keras_model,
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
        ))
    return tpu_model

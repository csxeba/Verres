import tensorflow as tf

import verres as V


def factory(config: V.Config, schedule) -> tf.keras.optimizers.Optimizer:
    spec = config.training.optimizer_spec.copy()
    optimizer_name = spec.pop("name", "Adam")
    optimizer_type = getattr(tf.keras.optimizers, optimizer_name, None)
    if optimizer_type is None:
        raise RuntimeError(f"Unsupported optimizer: {optimizer_name}")
    optimizer = optimizer_type(learning_rate=schedule, **spec)
    if config.context.verbose > 1:
        print(f" [Verres.optimizer] - Factory built: {optimizer_name}")

    return optimizer

import tensorflow as tf

import verres as V


class ConstantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = float(learning_rate)

    def __call__(self, step):
        return self.learning_rate

    def get_config(self):
        return dict(learning_rate=self.learning_rate)


def factory(spec: dict) -> tf.optimizers.schedules.LearningRateSchedule:

    spec = dict(spec)
    name = spec.pop("name", "default")

    if name.lower() in {"default", "constant"}:
        scheduler = ConstantSchedule(float(spec["learning_rate"]))
    else:
        scheduler_type = getattr(tf.optimizers.schedules, name, None)
        if scheduler_type is None:
            raise KeyError(f"No such scheduler: {name}")
        scheduler = scheduler_type(**spec)
    print(f" [Verres.schedule] - Factory built: {name}")

    return scheduler

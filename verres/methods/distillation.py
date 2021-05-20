from typing import Tuple

import numpy as np


DISTILLATION_MODE_SIMPLE = "simple"
DISTILLATION_MODE_COMBINED = "combined"


class TemperedSoftMax(Activation):

    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__("softmax")

    def call(self, inputs):
        return super().call(inputs / self.temperature, )


class Distillary:

    def __init__(self,
                 student: Model,
                 *teachers: Model,
                 mode=DISTILLATION_MODE_SIMPLE):

        self.student = student
        self.teachers = teachers  # type: Tuple[Model]
        self.mode = mode

    def _prepare_teacher(self):
        for teacher in self.teachers:
            dummy_input = np.zeros((1,) + teacher.input_shape[1:])
            teacher.predict(dummy_input)

    def _distillation_simple(self, batch):
        x, y = batch
        for teacher in self.teachers:
            y += teacher.predict(x)
        return x, y / len(self.teachers)

    def _distillation_combined(self, batch):
        for teacher in self.teachers:
            teacher.train_on_batch(*batch)
        return self._distillation_simple(batch)

    def fit_generator(self, generator,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0):

        self._prepare_teacher()
        kw = locals().copy()
        if "self" in kw:
            kw.pop("self")
        g = kw.pop("generator")
        if self.mode == DISTILLATION_MODE_SIMPLE:
            g = map(self._distillation_simple, g)
        elif self.mode == DISTILLATION_MODE_COMBINED:
            g = map(self._distillation_combined, g)
        else:
            raise ValueError("Unknown distillation mode: {}".format(self.mode))

        return self.student.fit_generator(g, **kw)

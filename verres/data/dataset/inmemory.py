import numpy as np
import tensorflow as tf

import verres as V
from .abstract import Dataset


class InMemoryImageClassification(Dataset):

    def unpack(self, ID):
        pass

    def __len__(self):
        pass

    def __init__(self, config: V.Config, spec: dict, subset: str = None):

        super().__init__(config)
        self.data = None
        self.labels = None
        self.training_indices = None
        self.validation_indices = None
        self.sample_ids = None
        self.input_shape = None
        self.num_classes = None
        self.sparse_label = onehot_label

        if self.data is None:
            self._load_data()

    def _load_data(self):
        module = {"cifar10": tf.keras.datasets.cifar10,
                  "cifar100": tf.keras.datasets.cifar10,
                  "mnist": tf.keras.datasets.mnist,
                  "fashion_mnist": tf.keras.datasets.fashion_mnist}[self.dataset]

        learning, validation = module.load_data()

        N_learning = len(learning[0])

        self.data = np.concatenate([learning[0], validation[0]])
        if self.data.ndim == 3:
            self.data = self.data[..., None]
        self.labels = np.concatenate([learning[1], validation[1]])
        self.labels = np.squeeze(self.labels)
        self.labels -= self.labels.min()
        self.num_classes = self.labels.max()+1
        self.input_shape = self.data.shape[1:]
        self.sample_ids = np.arange(len(self.data))
        self.training_indices = self.sample_ids[:N_learning]
        self.validation_indices = self.sample_ids[N_learning:]
        self.num_samples = {"train": len(self.training_indices), "val": len(self.validation_indices)}

    def get_ids(self, subset):
        return {"train": self.training_indices, "val": self.validation_indices}[subset]

    def steps_per_epoch(self, batch_size, subset):
        return self.num_samples[subset] // batch_size

    def load_input(self, sample_id):
        x = self.data[sample_id] / 255.
        if x.ndim == 2:
            x = x[..., None]
        return x

    def load_label(self, sample_id):
        y = self.labels[sample_id]
        if self.sparse_label:
            y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        return y


class CIFAR10(InMemoryImageClassification):
    ...


class CIFAR100(InMemoryImageClassification):
    ...


class MNIST(InMemoryImageClassification):
    ...


class FASHION_MNIST(InMemoryImageClassification):
    ...

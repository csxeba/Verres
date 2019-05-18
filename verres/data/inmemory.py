import numpy as np
from keras.datasets import cifar10 as _cifar10, cifar100 as _cifar100, mnist as _mnist, fashion_mnist as _fashion_mnist
from keras.utils import to_categorical


class _InMemoryImageClassificationDatasets:

    def __init__(self):

        self.dataset = None
        self.data = None
        self.labels = None
        self.training_indices = None
        self.validation_indices = None
        self.sample_ids = None

        if self.data is None:
            self._load_data()

    def _load_data(self):
        module = {"cifar10": _cifar10,
                  "cifar100": _cifar100,
                  "mnist": _mnist,
                  "fashion_mnist": _fashion_mnist}[self.dataset]

        learning, validation = module.load_data()

        N_learning = len(learning[0])

        self.data = np.concatenate([learning[0], validation[0]])
        self.labels = np.concatenate([learning[1], validation[1]])
        self.num_classes = self.labels.max()
        self.sample_ids = np.arange(len(self.data))
        self.training_indices = self.sample_ids[:N_learning]
        self.validation_indices = self.sample_ids[N_learning:]

    def load_input(self, sample_id):
        x = self.data[sample_id] / 255.
        if x.ndim == 2:
            x = x[..., None]
        return x

    def load_label(self, sample_id):
        return to_categorical(self.labels[sample_id])

    @property
    def dataset(self):
        return self.__class__.__name__.lower()

    def stream(self, batch_size, subset, shuffle=True):
        ids = {"train": self.training_indices, "val": self.validation_indices}[subset]
        while 1:
            X, Y = [], []
            if shuffle:
                np.random.shuffle(ids)
            for ID in ids:
                X.append(self.load_input(ID))
                Y.append(self.load_label(ID))
            X = np.array(X)
            Y = np.array(Y)
            yield X, Y


class CIFAR10(_InMemoryImageClassificationDatasets):
    ...


class CIFAR100(_InMemoryImageClassificationDatasets):
    ...


class MNIST(_InMemoryImageClassificationDatasets):
    ...


class FASHION_MNIST(_InMemoryImageClassificationDatasets):
    ...

"""PonderNet implementation according to https://arxiv.org/pdf/2107.05407.pdf"""

from collections import defaultdict, deque
from enum import Enum
from typing import NamedTuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfl

from verres.architecture.layers import block, ponder
from verres.architecture.types import IntermediateResult


class SubsetEnum(Enum):
    TRAIN = "train"
    TEST = "test"


class Batch(NamedTuple):
    x: tf.Tensor
    y: tf.Tensor


class MNIST(NamedTuple):
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    @classmethod
    def load_from_keras(cls):
        train_set, test_set = tf.keras.datasets.mnist.load_data()
        return cls(train_x=train_set[0], train_y=train_set[1],
                   test_x=test_set[0], test_y=test_set[1])

    def stream(self, batch_size: int, subset: SubsetEnum, shuffle: bool):
        x, y = {SubsetEnum.TRAIN: (self.train_x, self.train_y),
                SubsetEnum.TEST: (self.test_x, self.test_y)}[subset]

        index = np.arange(len(x))

        batch_x = np.empty((batch_size, x.shape[1], x.shape[2], 1), dtype=np.float32)
        batch_y = np.empty((batch_size,), dtype=int)

        while 1:
            if shuffle:
                np.random.shuffle(index)

            for i, idx in enumerate(index, start=0):
                counter = i % batch_size
                batch_x[counter] = x[i][..., None]
                batch_y[counter] = y[i]
                if counter == batch_size - 1:
                    yield Batch(x=tf.convert_to_tensor(batch_x), y=tf.convert_to_tensor(batch_y))


class ExperimentParameters(NamedTuple):
    ponder: bool
    ponder_prior_p: float
    batch_size: int
    epochs: int
    steps_per_epoch: int
    learning_rate: float
    ponder_loss_component_weight: float
    training_metrics_smoothing_window_size: int


class MetricAccumulator:

    def __init__(self, smoothing_window_size: int):
        self.accumulator = defaultdict(lambda: deque(maxlen=smoothing_window_size))

    def add(self, key, value):
        self.accumulator[key].append(value)

    def get(self):
        result = {}
        for key, value in self.accumulator.items():
            result[key] = np.mean(value)
        return result


class PonderVGG19(tf.keras.Model):

    def __init__(self,
                 pondernet: bool = True,
                 ponder_prior_p: float = 1. / 10.,
                 l2_reg: float = 0.):

        super().__init__()

        regularizer = tf.keras.regularizers.l2(l2_reg)

        self.layer_objects = [
            block.VRSConvolution(width=16, activation="relu", batch_normalize=False, name="block1_conv1",
                                 kernel_regularizer=regularizer),
            block.VRSConvolution(width=16, activation="relu", batch_normalize=False, name="block1_conv2",
                                 kernel_regularizer=regularizer),
            tfl.MaxPool2D(name="block1_pool")]

        if pondernet:
            self.ponder_layer = \
                ponder.PonderConvolution(width=32, activation="relu", batch_normalize=False, name="block2_conv1",
                                         kernel_regularizer=regularizer)
        else:
            self.ponder_layer = \
                block.VRSConvolution(width=32, activation="relu", batch_normalize=False, name="block2_conv1",
                                     kernel_regularizer=regularizer)

        self.classifier = tfl.Dense(10, activation="softmax", name="dense_classifier")

        self.is_pondernet = pondernet
        self.prior_p = tf.convert_to_tensor(ponder_prior_p)

    def call(self, x, **kwargs):
        x = x / 255.
        for layer in self.layer_objects:
            x = layer(x)
        im_result = self.ponder_layer(x)
        if self.is_pondernet:
            features = im_result.outputs["output"]
            ponder_loss = (tf.math.log(self.prior_p / im_result.outputs["lambda"])
                           + ((1. - self.prior_p) / self.prior_p)
                           * tf.math.log((1. - self.prior_p) / (1. - im_result.outputs["lambda"])))
            mean_steps = im_result.metrics["avg_steps"]
            probabilities = im_result.outputs["probs"]  # [ponder]

        else:
            features = im_result[None, ...]
            ponder_loss = 0.
            mean_steps = 1.
            probabilities = tf.ones(shape=[1], dtype=tf.float32)

        # GlobalAveragePooling of the spatial dimensions
        features = tf.reduce_mean(features, axis=(2, 3))  # [ponder, batch, channel] or [batch, channel]
        predictions = self.classifier(features)

        return IntermediateResult(outputs={"predictions": predictions, "probabilities": probabilities},
                                  metrics={"avg_steps": mean_steps},
                                  losses={"ponder_loss": ponder_loss})


def broadcasted_weighted_cross_entropy(y_true, y_pred, probs):
    onehots = tf.keras.utils.to_categorical(y_true, num_classes=10)[None, ...]  # shape: [1, batch, dim]
    cross_entropies = tf.reduce_mean(onehots * -tf.math.log(y_pred), axis=-1)
    return tf.reduce_mean(cross_entropies * probs)


def broadcasted_accuracy(y_true, y_pred):
    pred_classes = tf.argmax(y_pred, axis=2)  # shape: [ponder, batch]
    equalities = tf.expand_dims(y_true, axis=0) == pred_classes  # shape: [ponder, batch]
    accuracy = tf.reduce_mean(tf.cast(equalities, tf.float32), axis=1)
    return accuracy


def experiment(params: ExperimentParameters):
    data = MNIST.load_from_keras()
    model = PonderVGG19(pondernet=params.ponder, ponder_prior_p=params.ponder_prior_p)

    optimizer = tf.keras.optimizers.Adam(params.learning_rate)
    data_iterator = iter(data.stream(batch_size=params.batch_size, subset=SubsetEnum.TRAIN, shuffle=True))
    val_x = tf.convert_to_tensor(data.test_x[..., None], dtype=tf.float32)
    val_y = tf.convert_to_tensor(data.test_y)

    for epoch in range(1, params.epochs+1):
        print(f"\nEpoch {epoch}/{params.epochs}")
        metric_accumulator = MetricAccumulator(params.training_metrics_smoothing_window_size)
        for step in range(1, params.steps_per_epoch+1):
            batch = next(data_iterator)
            with tf.GradientTape() as tape:
                result = model(batch.x)
                softmaxes = result.outputs["predictions"]  # shape: [ponder, batch, num_classes]
                final_classification_loss = broadcasted_weighted_cross_entropy(
                    batch.y, result.outputs["predictions"], result.outputs["probabilities"])
                ponder_loss = tf.reduce_mean(result.losses["ponder_loss"])
                l2 = sum(model.losses)
                total_loss = (final_classification_loss
                              + tf.convert_to_tensor(params.ponder_loss_component_weight) * ponder_loss
                              + l2)

            grads = tape.gradient(total_loss, model.trainable_weights)

            for g in grads:
                tf.cond(tf.reduce_any(tf.math.is_nan(g)),
                        true_fn=lambda: tf.assert_equal(0., 1.),
                        false_fn=lambda: tf.no_op)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            accuracy = broadcasted_accuracy(batch.y, softmaxes)

            metric_accumulator.add("loss", total_loss)
            metric_accumulator.add("xent", final_classification_loss)
            metric_accumulator.add("ponder", ponder_loss)
            metric_accumulator.add("acc", accuracy)
            metric_accumulator.add("E_steps", result.metrics["avg_steps"])

            ponders = int(tf.shape(accuracy)[0])
            for i in range(1, ponders):
                metric_accumulator.add(f"step_{i}_acc", accuracy[i])

            metrics = metric_accumulator.get()

            logstr = f"\r - Step {step:>4}/{params.steps_per_epoch}"\
                     f" - loss: {metrics['loss']:.4f}"\
                     f" - xent: {metrics['xent']:.4f}"\
                     f" - ponder: {metrics['ponder']:.4f}"\
                     f" - E_steps: {metrics['E_steps']:.2f} - "
            logstr += " - ".join(f"step_{i}_acc: {metrics[f'step_{i}_acc']:>6.2%}" for i in range(1, ponders))

            print(logstr, end="")

            if step % (params.steps_per_epoch // 10) == 0:
                print()

        print()

        result = model(val_x)
        softmaxes = result.outputs["predictions"]
        final_classification_loss = broadcasted_weighted_cross_entropy(
            val_y, result.outputs["predictions"], result.outputs["probabilities"])
        ponder_loss = tf.reduce_mean(result.losses["ponder_loss"])
        total_loss = final_classification_loss + tf.convert_to_tensor(params.ponder_loss_component_weight) * ponder_loss
        accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(val_y, softmaxes))

        print(f" - Validation"
              f" - loss: {total_loss:.4f}"
              f" - xent: {final_classification_loss:.4f}"
              f" - ponder: {ponder_loss:.4f}"
              f" - acc: {accuracy:>6.2%}"
              f" - E_steps: {tf.reduce_mean(result.metrics['avg_steps']):>2}")


def main():
    params = ExperimentParameters(
        ponder=True,
        ponder_prior_p=1/3,
        batch_size=64,
        epochs=30,
        steps_per_epoch=1000,
        learning_rate=3e-4,
        ponder_loss_component_weight=0.01,
        training_metrics_smoothing_window_size=10
    )

    print(" PARAMS:", params)
    experiment(params)


if __name__ == '__main__':
    main()

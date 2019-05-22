from collections import deque

import numpy as np

from verres.optimizers import adabound
from verres.keras_engine import get_engine


class GAN:

    def __init__(self, latent_dim=32, ann_engine=None):
        self.engine = get_engine(ann_engine)
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None

    def build_baseline(self):

        z_input = self.engine.layers.Input(shape=(self.latent_dim,))

        x = self.engine.layers.Dense(1024)(z_input)
        x = self.engine.layers.Reshape((1, 1, 1024))(x)

        x = self.engine.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")(x)
        x = self.engine.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")(x)
        x = self.engine.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation="relu")(x)
        x = self.engine.layers.Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation="sigmoid")(x)

        self.generator = self.engine.Model(z_input, x, name="Generator")

        image = self.engine.layers.Input(shape=(64, 64, 3))

        x = self.engine.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(image)
        x = self.engine.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(x)
        x = self.engine.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(x)
        x = self.engine.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation="relu")(x)

        x = self.engine.layers.Flatten()(x)

        x = self.engine.layers.Dense(1, activation="sigmoid")(x)

        self.discriminator = self.engine.Model(image, x, name="Discriminator")
        self.gan = self.engine.Model(z_input, self.discriminator(self.generator(z_input)), name="GAN")

        self._disable_generator()
        self.discriminator.compile(optimizer=adabound.build(2e-4), loss="binary_crossentropy")
        self._disable_discriminator()
        self.gan.compile(optimizer=adabound.build(2e-4), loss="binary_crossentropy")
        self._enable()

    def build_reference(self):
        noise = self.engine.layers.Input((self.latent_dim,))
        x = self.engine.layers.Dense(1024, activation="relu", input_dim=self.latent_dim)(noise)
        x = self.engine.layers.Dense(128 * 8 * 8, activation="relu")(x)
        x = self.engine.layers.Reshape((8, 8, 128))(x)
        x = self.engine.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same")(x)  # 16
        x = self.engine.layers.BatchNormalization(momentum=0.8)(x)
        x = self.engine.layers.Activation("relu")(x)
        x = self.engine.layers.Conv2D(128, kernel_size=3, padding="same")(x)
        x = self.engine.layers.BatchNormalization(momentum=0.8)(x)
        x = self.engine.layers.Activation("relu")(x)
        x = self.engine.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)  # 32
        x = self.engine.layers.BatchNormalization(momentum=0.8)(x)
        x = self.engine.layers.Activation("relu")(x)
        x = self.engine.layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = self.engine.layers.BatchNormalization(momentum=0.8)(x)
        x = self.engine.layers.Activation("relu")(x)
        x = self.engine.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)  # 64
        x = self.engine.layers.BatchNormalization(momentum=0.8)(x)
        x = self.engine.layers.Activation("relu")(x)
        x = self.engine.layers.Conv2D(32, kernel_size=3, padding="same")(x)
        x = self.engine.layers.BatchNormalization()(x)
        x = self.engine.layers.Activation("relu")(x)
        x = self.engine.layers.Conv2D(3, kernel_size=3, padding="same")(x)
        x = self.engine.layers.Activation("sigmoid")(x)

        self.generator = self.engine.Model(noise, x, name="Generator")

        image = self.engine.layers.Input((64, 64, 3))

        x = self.engine.layers.Conv2D(32, kernel_size=3, strides=1, padding="same")(image)  # 32
        x = self.engine.layers.LeakyReLU(alpha=0.2)(x)
        x = self.engine.layers.Conv2D(32, kernel_size=4, strides=2, padding="same")(x)  # 32
        x = self.engine.layers.LeakyReLU(alpha=0.2)(x)
        x = self.engine.layers.Conv2D(64, kernel_size=3, strides=1, padding="same")(x)  # 16
        x = self.engine.layers.LeakyReLU(alpha=0.2)(x)
        x = self.engine.layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(x)  # 16
        x = self.engine.layers.LeakyReLU(alpha=0.2)(x)
        x = self.engine.layers.Conv2D(128, kernel_size=3, strides=1, padding="same")(x)  # 8
        x = self.engine.layers.LeakyReLU(alpha=0.2)(x)
        x = self.engine.layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)  # 8
        x = self.engine.layers.LeakyReLU(alpha=0.2)(x)
        x = self.engine.layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)  # 4
        x = self.engine.layers.LeakyReLU(alpha=0.2)(x)
        x = self.engine.layers.Flatten()(x)
        # x = Dropout(0.25)(x)
        x = self.engine.layers.Dense(256, activation="relu")(x)
        # x = Dropout(0.25)(x)
        x = self.engine.layers.Dense(1, activation='sigmoid')(x)

        self.discriminator = self.engine.Model(image, x, name="Discriminator")
        self.gan = self.engine.Model(noise, self.discriminator(self.generator(noise)), name="GAN")

        self._disable_generator()
        self.discriminator.compile(optimizer=adabound.build(2e-4, beta_1=0.5), loss="mse")
        self._disable_discriminator()
        self.gan.compile(optimizer=adabound.build(2e-4, beta_1=0.5), loss="mse")
        self._enable()

    @staticmethod
    def _wgan_loss(y, o):
        return y * o

    def _disable_generator(self):
        self.generator.trainable = False
        self.discriminator.trainable = True

    def _disable_discriminator(self):
        self.generator.trainable = True
        self.discriminator.trainable = False

    def _enable(self):
        self.generator.trainable = True
        self.discriminator.trainable = True

    def train_discriminator(self, images, m):
        self._disable_generator()

        z = np.random.randn(m, self.latent_dim)

        fakes = self.generator.predict(z)
        inputs = np.concatenate([images, fakes])

        trues = np.ones(m)
        falses = np.zeros(m)

        labels = np.concatenate([trues, falses])
        loss = self.discriminator.train_on_batch(inputs, labels)

        return loss

    def train_generator(self, m, repeat=1):
        self._disable_discriminator()
        losses = []

        for r in range(repeat):
            z = np.random.randn(m, self.latent_dim)
            trues = np.ones(m)
            losses.append(self.gan.train_on_batch(z, trues))

        return np.mean(losses)

    def fit_generator(self, data_generator, updates_per_epoch, epochs, verbose=1,
                      generator_num_repeats=3,
                      latest_checkpoint_name=None,
                      epoch_checkpoint_name=None,
                      epoch_checkpoint_period=None):

        ustrlen = len(str(updates_per_epoch))
        estrlen = len(str(epochs))
        dlosses = deque(maxlen=10)
        glosses = deque(maxlen=10)
        count = 1
        for epoch in range(1, epochs+1):
            for update, (images, _) in enumerate(data_generator, start=1):
                m = len(images)
                dloss = self.train_discriminator(images, m)
                gloss = self.train_generator(m, generator_num_repeats)
                dlosses.append(dloss)
                glosses.append(gloss)
                if verbose:
                    print("\rEpoch {:>{e}}/{} Update {:>{u}}/{} DLOSS: {:>7.4f} | GLOSS: {:>7.4f}"
                          .format(epoch, epochs, update, updates_per_epoch,
                                  np.mean(dloss), np.mean(gloss),
                                  u=ustrlen, e=estrlen), end="")
                    if count % 10 == 0:
                        print()
                count += 1
                if update >= updates_per_epoch:
                    break
            if verbose:
                print()
            if epoch_checkpoint_name is not None:
                epoch_checkpoint_period = epoch_checkpoint_period or 1
                if epoch % epoch_checkpoint_period == 0:
                    self.gan.save(epoch_checkpoint_name.format(epoch=epoch))
            if latest_checkpoint_name is not None:
                self.gan.save(latest_checkpoint_name)

        self._enable()

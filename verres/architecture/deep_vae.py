import cv2

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, LeakyReLU, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose, Add
from keras import backend as K


def block(block_input, num_filters, depth):
    skip = Conv2D(num_filters, (1, 1))(block_input)
    skip = LeakyReLU()(skip)
    x = Conv2D(num_filters, (3, 3), padding="same")(skip)
    x = LeakyReLU()(x)
    for d in range(depth-1):
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
    x = Add()([skip, x])
    return x


def ds(inputs, num_filters):
    x = Conv2D(num_filters, (4, 4), strides=2)(inputs)
    x = LeakyReLU()(x)
    return x


class VAE:

    def __init__(self, latent_dim, beta=1., loss="mse"):
        self.latent_dim = latent_dim
        self._loss = loss
        self.beta = beta
        self.encoder = None
        self.decoder = None
        self.vae = None

    def somnium_architecture(self):
        # Encoder Part

        vae_x = Input(shape=(64, 64, 3))
        vae_c1 = Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(vae_x)
        vae_c2 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(vae_c1)
        vae_c3 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(vae_c2)
        vae_c4 = Conv2D(filters=128, kernel_size=4, strides=2, activation="relu")(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(self.latent_dim)(vae_z_in)
        vae_z_log_var = Dense(self.latent_dim)(vae_z_in)

        vae_z = Lambda(self._sampling)([vae_z_mean, vae_z_log_var])

        # Decoder part

        vae_z_input = Input(shape=(self.latent_dim,))

        vae_dense = Dense(64*5*5, activation="relu")(vae_z_input)
        vae_z_out = Reshape((5, 5, 64))(vae_dense)

        vae_d2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")(vae_z_out)
        vae_d3 = Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation="relu")(vae_d2)
        vae_d4 = Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation="sigmoid")(vae_d3)

        # Constructed Models

        vae_encoder = Model(vae_x, vae_z_mean)
        vae_sampler = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4)
        vae = Model(vae_x, vae_decoder(vae_sampler(vae_x)))

        # Custom Loss Functions

        def reconstruction(y_true, y_pred):
            if self._loss == "mse":
                r = K.square(y_true - y_pred)
            else:
                r = -(y_true * K.log(y_pred) + (1. - y_true) * K.log(1. - y_pred))
            return K.sum(r, axis=(1, 2, 3))

        def kl_divergence(y_true=None, y_pred=None):
            return - 0.5 * K.sum(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=1)

        def vae_loss(y_true, y_pred):
            r_loss = reconstruction(y_true, y_pred)
            kl_loss = kl_divergence()
            return K.mean(r_loss + self.beta * kl_loss)

        vae.compile(optimizer="adam", loss=vae_loss, metrics=[reconstruction, kl_divergence])

        self.vae = vae
        self.encoder = vae_encoder
        self.decoder = vae_decoder

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    @staticmethod
    def preprocess(frame):
        ds = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_CUBIC)
        inputs = ds[None, ...] / 255.
        return inputs

    @staticmethod
    def deprocess(output, original_shape):
        depro = (output[0] * 255.).astype("uint8")
        result = cv2.resize(depro, original_shape, interpolation=cv2.INTER_CUBIC)
        return result

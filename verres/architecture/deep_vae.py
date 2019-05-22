import cv2

from verres.keras_engine import get_engine


class VAE:

    def __init__(self, latent_dim, beta=1., loss="mse", ann_engine=None):
        self.engine = get_engine(ann_engine)
        self.latent_dim = latent_dim
        self._loss = loss
        self.beta = beta
        self.encoder = None
        self.decoder = None
        self.vae = None

    def somnium_architecture(self):
        # Encoder Part

        vae_x = self.engine.layers.Input(shape=(64, 64, 3))
        vae_c1 = self.engine.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(vae_x)
        vae_c2 = self.engine.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(vae_c1)
        vae_c3 = self.engine.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(vae_c2)
        vae_c4 = self.engine.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation="relu")(vae_c3)

        vae_z_in = self.engine.layers.Flatten()(vae_c4)

        vae_z_mean = self.engine.layers.Dense(self.latent_dim)(vae_z_in)
        vae_z_log_var = self.engine.layers.Dense(self.latent_dim)(vae_z_in)

        vae_z = self.engine.layers.Lambda(self._sampling)([vae_z_mean, vae_z_log_var])

        # Decoder part

        vae_z_input = self.engine.layers.Input(shape=(self.latent_dim,))

        vae_dense = self.engine.layers.Dense(64*5*5, activation="relu")(vae_z_input)
        vae_z_out = self.engine.layers.Reshape((5, 5, 64))(vae_dense)

        vae_d2 = self.engine.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")(vae_z_out)
        vae_d3 = self.engine.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation="relu")(vae_d2)
        vae_d4 = self.engine.layers.Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation="sigmoid")(vae_d3)

        # Constructed Models

        vae_encoder = self.engine.Model(vae_x, vae_z_mean)
        vae_sampler = self.engine.Model(vae_x, vae_z)
        vae_decoder = self.engine.Model(vae_z_input, vae_d4)
        vae = self.engine.Model(vae_x, vae_decoder(vae_sampler(vae_x)))

        # Custom Loss Functions

        def reconstruction(y_true, y_pred):
            if self._loss == "mse":
                r = self.engine.backend.square(y_true - y_pred)
            else:
                r = -(y_true * self.engine.backend.log(y_pred) + (1. - y_true) * self.engine.backend.log(1. - y_pred))
            return self.engine.backend.sum(r, axis=(1, 2, 3))

        def kl_divergence(y_true=None, y_pred=None):
            z = 1 + vae_z_log_var - self.engine.backend.square(vae_z_mean) - self.engine.backend.exp(vae_z_log_var)
            return - 0.5 * self.engine.backend.sum(z, axis=1)

        def vae_loss(y_true, y_pred):
            r_loss = reconstruction(y_true, y_pred)
            kl_loss = kl_divergence()
            return self.engine.backend.mean(r_loss + self.beta * kl_loss)

        vae.compile(optimizer="adam", loss=vae_loss, metrics=[reconstruction, kl_divergence])

        self.vae = vae
        self.encoder = vae_encoder
        self.decoder = vae_decoder

    def _sampling(self, args):
        z_mean, z_log_var = args
        z_shape = (self.engine.backend.shape(z_mean)[0], self.latent_dim)
        epsilon = self.engine.backend.random_normal(shape=z_shape, mean=0., stddev=1.)
        return z_mean + self.engine.backend.exp(z_log_var / 2) * epsilon

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

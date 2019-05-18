import os

import numpy as np

from .alicegen import AliceLoader


class AliceImageStream:

    def __init__(self, data: AliceLoader, resize_shape=(64, 64)):
        self.data = data
        self.resize_shape = resize_shape

    def create_x(self, file_names):
        X = np.empty((len(file_names), self.resize_shape[0], self.resize_shape[1], 3), dtype="float32")
        for i, file in enumerate(file_names):
            x = self.data.load_and_resize(file, self.resize_shape, to_grey=False)
            X[i] = x / 255.
        return X

    def training_stream(self, batch_size=32):
        files = []
        for sequence in self.data.train_sequences.values():
            files.extend(list(sequence.values()))
        files = np.array(files)
        while 1:
            batch = np.random.choice(files, size=batch_size)
            X = self.create_x(batch)
            yield X, X

    def validation_set(self):
        files = sorted(list(self.data.val_sequence.values()))
        X = self.create_x(files)
        return X, X


class AliceDiffStream:

    def __init__(self, data_root):
        self.data_root = data_root
        self.train_files = {}
        self.val_files = {}
        self.train_y = np.load(os.path.join(data_root, "train_y.npy"))
        self.val_y = np.load(os.path.join(data_root, "val_y.npy"))
        self._fill_indices()
        self.N = len(self.train_y)

    def _fill_indices(self):
        subsets = {"train": self.train_files, "val": self.val_files}
        for file in os.listdir(self.data_root):
            filename = os.path.splitext(file)[0]
            subset, index = filename.split("_")
            if index in "xy":
                continue
            subsets[subset][int(index)] = os.path.join(self.data_root, file)

    @classmethod
    def generate_data(cls, data: AliceLoader, output_directory, encoder):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        image_stream = AliceImageStream(data)
        train_latents = []
        val_latents = []
        pairs = []
        for sequence in data.train_sequences.values():
            file_names = [sequence[frame_id] for frame_id in sorted(sequence)]
            pairs.extend(list(zip(file_names[:-1], file_names[1:])))
        N = len(pairs)
        for i, pair in enumerate(pairs):
            print(f"\rGenerating training data: {i/N:>7.2%} {N}/{i}", end="")
            encoder_inputs = image_stream.create_x(pair)
            latent_start, latent_end = encoder.predict(encoder_inputs)
            train_latents.append(latent_end - latent_start)
            start = data.load_and_resize(pair[0], shape=(32, 32), to_grey=True) / 255.
            end = data.load_and_resize(pair[1], shape=(32, 32), to_grey=True) / 255.
            D = (end - start).astype("float32")
            path = os.path.join(output_directory, f"train_{i}.npy")
            np.save(path, D)
        print()

        val_files = list(data.val_sequence.values())
        N = len(val_files) - 1
        for i, pair in enumerate(zip(val_files[:-1], val_files[1:])):
            print(f"\rGenerating validation data: {i / N:>7.2%} {N}/{i}", end="")
            encoder_inputs = image_stream.create_x(pair)
            latent_start, latent_end = encoder.predict(encoder_inputs)
            val_latents.append(latent_end - latent_start)
            start = data.load_and_resize(pair[0], shape=(32, 32), to_grey=True) / 255.
            end = data.load_and_resize(pair[1], shape=(32, 32), to_grey=True) / 255.
            D = (end - start).astype("float32")
            path = os.path.join(output_directory, f"val_{i}.npy")
            np.save(path, D)
        print()

        np.save(os.path.join(output_directory, "train_y.npy"), np.array(train_latents))
        np.save(os.path.join(output_directory, "val_y.npy"), np.array(val_latents))

        return cls(output_directory)

    def training_stream(self, batch_size=32):
        arg = np.array(list(self.train_files.keys()))
        while 1:
            idx = np.random.choice(arg, size=batch_size)
            X = np.array([np.load(self.train_files[i]) for i in idx])
            Y = self.train_y[idx]
            yield X, Y

    def validation_set(self):
        X = np.array([np.load(self.val_files[i]) for i in sorted(self.val_files)])
        return X, self.val_y

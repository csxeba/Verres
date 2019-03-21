import os
from collections import defaultdict

import numpy as np
import cv2
from keras.datasets import cifar10
from keras.utils import to_categorical


def load_cifar():
    (lX, lY), (tX, tY) = cifar10.load_data()
    lX, tX = map(lambda x: x / 255., [lX, tX])
    lY, tY = map(to_categorical, [lY, tY])
    return lX, lY, tX, tY


def load_mnist():
    from keras.datasets import mnist
    from keras.utils import to_categorical

    (lX, lY), (vX, vY) = mnist.load_data()
    lX, vX = lX[..., None] / 255., vX[..., None] / 255.
    lY, vY = map(to_categorical, [lY, vY])
    return lX, lY, vX, vY


def stream_cifar(batch_size=32, data=None, batch_preprocessor=None):
    if data is None:
        data = load_cifar()
    lX, lY, *_ = data
    N = len(lX)
    arg = np.arange(N)
    while 1:
        np.random.shuffle(arg)
        for start in range(0, N, batch_size):
            idx = arg[start:start+batch_size]
            batch = lX[idx], lY[idx]
            if batch_preprocessor is not None:
                batch = batch_preprocessor(batch)
            yield batch


class AliceData:

    def __init__(self, vid_root, img_root):
        self.vid_root = vid_root
        self.img_root = img_root
        self.train_sequences = defaultdict(dict)
        self._fill_seq()
        self.val_sequence = self.train_sequences.pop("0")

    @property
    def N(self):
        return sum(len(sequence) for sequence in self.train_sequences.values())

    def _fill_seq(self):
        for img in os.listdir(self.img_root):
            name, ext = os.path.splitext(img)
            seq_id, frame_id = name.split("_")
            self.train_sequences[seq_id][frame_id] = img

    @classmethod
    def generate_images(cls, vid_root, img_root):
        vids = os.listdir(vid_root)
        for seq_id, file in enumerate(vids):
            print("\nParsing", file)
            reader = cv2.VideoCapture(vid_root + file)
            frame_id = 0
            while 1:
                success, frame = reader.read()
                if not success:
                    break
                filename = "{}{}_{}.jpg".format(img_root, seq_id, frame_id)
                cv2.imwrite(filename, frame)
                frame_id += 1
                print("\rWritten {}".format(frame_id), end="")

        print("Done!")
        return cls(vid_root, img_root)

    def load_and_resize(self, file_name, shape=None, to_grey=False):
        path = os.path.join(self.img_root, file_name)
        img = cv2.imread(path)
        if shape is not None:
            img = cv2.resize(img, shape[::-1], interpolation=cv2.INTER_CUBIC)
        if to_grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

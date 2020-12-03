import os

import numpy as np
import cv2


class PairStream:

    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise RuntimeError(f"No such file: {file_path}")
        self.file_path = file_path
        self.reader: cv2.VideoCapture = None

    def __enter__(self):
        if self.reader is None:
            self.reader = cv2.VideoCapture(self.file_path)
        return self

    def generator(self):
        last = self._read_frame()
        while 1:
            nxt = self._read_frame()
            yield {"last": last, "next": nxt}
            last = nxt

    def __iter__(self):
        return self.generator()

    def _read_frame(self):
        if self.reader is None:
            raise RuntimeError("Iteration must be executed in a with context.")
        success, frame = self.reader.read()
        if not success:
            raise RuntimeError(f"Couldn't read file: {self.file_path}")
        return frame.astype("int16")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.release()
        self.reader = None


class DiffStream(PairStream):

    def generator(self):
        for data in super().generator():
            data["dif"] = data["last"] - data["next"]
            yield data


class EventStream(DiffStream):

    def __init__(self, file_path, normed_difference_threshold: float = 0., sparse: bool = True):
        super().__init__(file_path)
        self.normed_difference_threshold = normed_difference_threshold
        self.sparse = sparse

    def generator(self):
        for data in super().generator():
            ndif = (data["dif"] / 255.).astype("float32")  # [-1 ... +1]
            if self.sparse:
                mask = np.abs(ndif) > self.normed_difference_threshold
                data["event_locations"] = np.argwhere(mask)
                data["event_values"] = ndif[mask]
                yield data
            else:
                ndif[np.abs(ndif) < self.normed_difference_threshold] = 0
                data["events"] = ndif
                yield data

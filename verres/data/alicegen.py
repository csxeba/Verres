import os
from collections import defaultdict

import cv2


class AliceLoader:

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

import tensorflow as tf

from verres.utils import masking


def make_images(paths):
    images = tf.io.read_file(paths)
    images = tf.image.decode_image(images)
    images = images / 255.
    images = tf.cast(images, tf.float32)
    return images


def make_masks(annotations):
    masks = []
    for anno in annotations:
        masks

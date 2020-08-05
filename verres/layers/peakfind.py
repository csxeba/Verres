import tensorflow as tf


class Peakfinder(tf.keras.Model):

    def __init__(self,
                 kernel_size: int = 5,
                 peak_nms_threshold: float = 0.01):

        super().__init__()
        self.pooler = tf.keras.layers.MaxPool2D(pool_size=(kernel_size, kernel_size),
                                                strides=(1, 1),
                                                padding="same")
        self.threshold = peak_nms_threshold

    @tf.function
    def call(self, heatmaps, training=None, mask=None):
        max_filtered = self.pooler(heatmaps)
        peaks = tf.where(max_filtered == heatmaps)
        scores = tf.gather(heatmaps, peaks)
        valid_score = scores > self.threshold
        return peaks[valid_score], scores[valid_score]

import os

import tensorflow as tf

from .loader import COCODoomLoader
from verres.tf_arch import vision
from verres.utils import box, visualize


class PredictionVisualizer:

    def __init__(self,
                 output_to_screen: bool = True,
                 output_file: str = None):

        self.device = visualize.output_device_factory(fps=30, scale=4, to_screen=output_to_screen,
                                                      output_file=output_file)
        self.visualizer = visualize.Visualizer()

    def draw_detection(self, image, model_output, alpha=0.5, write: bool = True):
        detection = tuple(map(lambda ar: ar.numpy(), model_output))
        boxes = box.convert_box_representation(
            detection[0], detection[1], detection[2])
        canvas = self.visualizer.overlay_boxes(image=image, boxes=boxes, stride=1, alpha=alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_raw_heatmap(self, image, model_output, alpha=0.5, write: bool = True):
        canvas = self.visualizer.overlay_heatmap(image=image, hmap=model_output[0], alpha=alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_raw_box(self, image, model_output, alpha=1, write: bool = True):
        canvas = self.visualizer.overlay_box_tensor(image, model_output[2], alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def __enter__(self):
        self.device.__enter__()
        return self

    def __exit__(self, *args):
        self.device.__exit__(*args)


class Mode:

    DETECTION = "detection"
    RAW_HEATMAP = "raw_heatmap"
    RAW_BOX = "raw_box"


def run_od(loader: COCODoomLoader,
           model: vision.ObjectDetector,
           mode: str = Mode.DETECTION,
           to_screen: bool = True,
           output_file: str = None,
           stop_after: int = None,
           alpha: float = 0.5):

    @tf.function
    def preprocess(image_path):
        data = tf.io.read_file(image_path)
        image = tf.io.decode_image(data)
        image = tf.image.convert_image_dtype(image, tf.float32)[..., ::-1]
        return image[None, ...]

    image_paths = [os.path.join(loader.cfg.images_root, meta["file_name"])
                   for meta in sorted(loader.image_meta.values(), key=lambda m: m["id"])]
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess, num_parallel_calls=3)
    dataset = dataset.prefetch(10)

    total = len(image_paths)
    if stop_after is None:
        stop_after = total

    with PredictionVisualizer(to_screen, output_file) as vis:

        for i, tensor in enumerate(dataset, start=1):

            if mode == Mode.DETECTION:
                output = model.detect(tensor)
                vis.draw_detection(tensor, output, alpha, write=True)

            else:
                output = model(tensor)

                if mode == Mode.RAW_HEATMAP:
                    vis.draw_raw_heatmap(tensor, output, alpha, write=True)
                elif mode == Mode.RAW_BOX:
                    vis.draw_raw_box(tensor, output, alpha, write=True)
                else:
                    raise NotImplementedError(f"Mode `{mode}` is not implemented!")

            if i >= stop_after:
                break

            print(f"\r [Verres] - Inference P: {i / stop_after:>7.2%}", end="")

    print()

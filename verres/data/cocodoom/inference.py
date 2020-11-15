import os
from typing import Union

import tensorflow as tf

from .loader import COCODoomLoader
from verres.tf_arch import vision
from verres.utils import box, visualize, profiling


class PredictionVisualizer:

    def __init__(self,
                 output_to_screen: bool = True,
                 output_file: str = None,
                 fps: int = 25,
                 scale: float = 4.):

        self.device = visualize.output_device_factory(fps=fps, scale=scale, to_screen=output_to_screen,
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

    def draw_raw_box(self, image, model_output, alpha=1., write: bool = True):
        canvas = self.visualizer.overlay_box_tensor(image, model_output[2], alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_semantic_segmentation(self, image, model_output, alpha=0.3, write: bool = True):
        canvas = self.visualizer.overlay_segmentation_mask(image, model_output[3], alpha=alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_instance_segmentation(self, image, model_output, alpha=0.3, write: bool = True):
        canvas = self.visualizer.overlay_instance_mask(image, model_output[2], alpha)
        if write:
            self.device.write(canvas)
        return canvas

    def draw_panoptic_segmentation(self, image, model_output, alpha=0.3, write: bool = True):
        canvas = self.visualizer.overlay_panoptic(image, coords=model_output[0], affils=model_output[1], alpha=alpha)
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
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"


def _run_common(model: Union[vision.ObjectDetector, vision.PanopticSegmentor],
                dataset: tf.data.Dataset,
                total: int,
                mode: str,
                to_screen: bool,
                output_file: str,
                stop_after: int,
                alpha: float,
                fps: int,
                scale: float):

    if stop_after is None:
        stop_after = total

    timer = profiling.MultiTimer()

    iterator = iter(dataset)

    with PredictionVisualizer(to_screen, output_file, fps, scale) as vis:

        for i in range(1, stop_after+1):

            with timer.time("data"):
                tensor = next(iterator)

            with timer.time("model"):
                if mode == Mode.DETECTION:
                    output = model.detect(tensor)
                elif mode == Mode.PANOPTIC:
                    output = model.detect(tensor)
                else:
                    output = model(tensor)

            with timer.time("visualizer"):
                if mode == Mode.RAW_HEATMAP:
                    vis.draw_raw_heatmap(tensor, output, alpha, write=True)
                elif mode == Mode.RAW_BOX:
                    vis.draw_raw_box(tensor, output, alpha, write=True)
                elif mode == Mode.DETECTION:
                    vis.draw_detection(tensor, output, alpha, write=True)
                elif mode == Mode.PANOPTIC:
                    vis.draw_panoptic_segmentation(tensor, output, alpha, write=True)
                elif mode == Mode.SEMANTIC:
                    vis.draw_semantic_segmentation(tensor, output, alpha, write=True)
                elif mode == Mode.INSTANCE:
                    vis.draw_instance_segmentation(tensor, output, alpha, write=True)
                else:
                    raise NotImplementedError(f"Mode `{mode}` is not implemented!")

            if i >= stop_after:
                break

            logstr = (
                    f"\r [Verres] - Inference P: {i / stop_after:>7.2%} - " +
                    " - ".join(f"{k}: {1/v:.4f} FPS" for k, v in timer.get_results(reset=True).items()))
            print(logstr, end="")

    print()


def run(loader: COCODoomLoader,
        model: Union[vision.ObjectDetector, vision.PanopticSegmentor],
        mode: str = Mode.DETECTION,
        to_screen: bool = True,
        output_file: str = None,
        stop_after: int = None,
        alpha: float = 0.5,
        fps: int = 25,
        scale: float = 4.):

    @tf.function
    def preprocess(image_path):
        data = tf.io.read_file(image_path)
        image = tf.io.decode_image(data)
        image = tf.image.convert_image_dtype(image, tf.float32)[..., ::-1]
        return image[None, ...]

    image_paths = [os.path.join(loader.cfg.images_root, meta["file_name"])
                   for meta in sorted(loader.image_meta.values(), key=lambda m: m["id"])]
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(10)

    total = len(image_paths)

    _run_common(model, dataset, total, mode, to_screen, output_file, stop_after, alpha, fps, scale)

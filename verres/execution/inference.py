from typing import List

import verres as V


class Mode:

    DETECTION = "detection"
    RAW_HEATMAP = "raw_heatmap"
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"


class InferenceExecutor:

    def __init__(self,
                 config: V.Config,
                 model: V.architecture.VRSArchitecture,
                 pipelines: List[V.data.Pipeline]):

        self.cfg = config
        self.model = model
        self.pipelines = pipelines
        self.codec = pipelines[0].codec

    @classmethod
    def factory(cls, config: V.Config):
        model = V.architecture.VRSArchitecture.factory(config)
        pipeline = V.data.factory(config, specs=config.inference.data)
        return cls(config, model, pipeline)

    def execute(self):

        timer = V.utils.profiling.MultiTimer()
        iterator = iter(V.data.streaming.get_tf_dataset(self.cfg, self.pipelines, shuffle=False, batch_size=1))

        mode = self.cfg.inference.visualization_mode

        with V.visualization.PredictionVisualizer(self.cfg) as vis:

            for i in range(1, self.cfg.inference.total_num_frames + 1):

                with timer.time("data"):
                    tensor = next(iterator)["image"]

                with timer.time("model"):
                    output = self.model(self.model.preprocess_input(tensor), training=False)
                    if mode == Mode.DETECTION:
                        output = self.codec.decode({k: v[0] for k, v in output.items()})

                with timer.time("visualizer"):
                    if mode == Mode.RAW_HEATMAP:
                        vis.draw_raw_heatmap(tensor, output, 0.4, write=True)
                    elif mode == Mode.DETECTION:
                        vis.draw_detection(tensor, output.object_keypoint_coords, output.object_types,
                                           0.4, write=True)
                    elif mode == Mode.PANOPTIC:
                        vis.draw_panoptic_segmentation(tensor, output, 0.4, write=True)
                    elif mode == Mode.SEMANTIC:
                        vis.draw_semantic_segmentation(tensor, output, 0.4, write=True)
                    elif mode == Mode.INSTANCE:
                        vis.draw_instance_segmentation(tensor, output, 0.4, write=True)
                    else:
                        raise NotImplementedError(f"Mode `{mode}` is not implemented!")

                logstr = (
                        f"\r [Verres] - Inference P: {i / self.cfg.inference.total_num_frames:>7.2%} - " +
                        " - ".join(f"{k}: {1/v:.4f} FPS" for k, v in timer.get_results(reset=True).items()))
                print(logstr, end="")

                if i >= self.cfg.inference.total_num_frames:
                    break

        print()

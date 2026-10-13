import pathlib
import time
from typing import List, Union

import cv2

# MP4 container is the default because DIVX/AVI files are unplayable in most browsers.
_FOURCC_BY_EXTENSION = {".mp4": "mp4v", ".m4v": "mp4v", ".mov": "mp4v", ".avi": "XVID"}


class OutputDevice:

    def __init__(self, scale: float, fps: int):
        self.scale = scale
        self.fps = fps

    def write(self, frame):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class CV2Screen(OutputDevice):

    def __init__(self, window_name="CV2Screen", fps=None, scale=1., realtime_playback=False):
        super().__init__(scale, fps)
        self.name = window_name
        # Playback throttling is opt-in: by default the screen must not limit inference throughput.
        self.sleep_per_frame = 1. / fps if realtime_playback and fps and fps > 0 else 0.
        self.online = False

    def write(self, frame):
        if not self.online:
            self.online = True
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(self.name, frame)
        cv2.waitKey(1)  # only pumps the GUI event loop, it does not throttle
        if self.sleep_per_frame:
            time.sleep(self.sleep_per_frame)

    def teardown(self):
        if self.online:
            cv2.destroyWindow(self.name)
        self.online = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()

    def __del__(self):
        self.teardown()


class CV2VideoWriter(OutputDevice):

    def __init__(self, file_name: str, fps: int, scale: float = 1.):
        super().__init__(scale, fps)
        if not fps or fps <= 0:
            raise RuntimeError(f"Video output needs a positive fps, got {fps}. Set `inference.fps`.")
        path = pathlib.Path(file_name)
        if not path.suffix:
            path = path.with_suffix(".mp4")
        fourcc_name = _FOURCC_BY_EXTENSION.get(path.suffix.lower())
        if fourcc_name is None:
            raise RuntimeError(f"Unsupported video extension `{path.suffix}`. "
                               f"Use one of: {', '.join(sorted(_FOURCC_BY_EXTENSION))}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file_name = str(path)
        self.fourcc_name = fourcc_name
        self.device: Union[cv2.VideoWriter, None] = None

    def write(self, frame):
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        if self.device is None:
            height, width = frame.shape[:2]
            # Frame size is only known at the first frame, so the writer is opened lazily.
            self.device = cv2.VideoWriter(
                self.file_name,
                cv2.VideoWriter_fourcc(*self.fourcc_name),
                float(self.fps),
                (width, height),
            )
            if not self.device.isOpened():
                raise RuntimeError(f"Could not open `{self.file_name}` with codec `{self.fourcc_name}`")
        self.device.write(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device is not None:
            self.device.release()
            self.device = None


class OutputDeviceList:

    def __init__(self,
                 output_device_list: List[OutputDevice],
                 scale: float = None):

        self.devices: List[OutputDevice] = output_device_list
        if not self.devices:
            raise RuntimeError("No devices!")
        self.in_context = False
        if scale is not None:
            for device in self.devices:
                if device.scale not in [scale, 1.]:
                    raise RuntimeError("Ambiguous definitions for scale!")
                device.scale = 1.
        self.scale = scale

    def __enter__(self):
        self.in_context = True
        for device in self.devices:
            device.__enter__()
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.in_context = False
        for device in self.devices:
            device.__exit__(exc_type, exc_val, exc_tb)

    def write(self, frame):
        if not self.in_context:
            raise RuntimeError("Please execute write() in a context manager!")
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        for device in self.devices:
            device.write(frame)


def output_device_factory(fps: int,
                          scale: float = 1.,
                          to_screen: bool = True,
                          output_file: str = None,
                          realtime_playback: bool = False):

    devices = []
    if to_screen:
        devices.append(CV2Screen(fps=fps, scale=1., realtime_playback=realtime_playback))
    if output_file:
        devices.append(CV2VideoWriter(str(output_file), fps=fps, scale=1.))
    if not devices:
        raise RuntimeError("No output device requested: enable inference.to_screen or set a video path.")

    return OutputDeviceList(devices, scale)

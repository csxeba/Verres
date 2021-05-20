import pathlib
from typing import Union, List

import cv2


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

    def __init__(self, window_name="CV2Screen", fps=None, scale=1.):
        super().__init__(scale, fps)
        self.name = window_name
        if self.fps is None:
            self.fps = 1000
        self.spf = 1000 // fps
        self.online = False

    def write(self, frame):
        if not self.online:
            self.online = True
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(self.name, frame)
        cv2.waitKey(self.spf)

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

    def __init__(self, file_name: str, fps: int, scale: float):
        super().__init__(scale, fps)
        self.file_name = file_name
        self.size = None
        self.device: Union[cv2.VideoWriter, None] = None
        self._in_context = False

    def __enter__(self):
        self._in_context = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device is not None:
            self.device.release()
            self.device = None
            self._in_context = False

    def write(self, frame):
        if not self._in_context:
            raise RuntimeError("Please run in a `with` context!")
        if self.device is None:
            fourcc = cv2.VideoWriter_fourcc(*"DIVX")
            self.device = cv2.VideoWriter(self.file_name, fourcc, float(self.fps), frame.shape[:2][::-1])
        if self.scale != 1:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        self.device.write(frame)


class OutputDeviceList:

    def __init__(self,
                 output_device_list: List[OutputDevice],
                 scale: float = None):

        self.devices: List[OutputDevice] = output_device_list
        self.in_context = False
        if scale is not None:
            for device in self.devices:
                if device.scale not in [scale, 1.]:
                    raise RuntimeError("Ambiguous definitions for scale!")
                device.scale = 1.
        self.scale = scale
        if not self.devices:
            raise RuntimeError("No devices!")

    def __enter__(self):
        self.in_context = True
        for device in self.devices:
            device.__enter__()

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
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
                          output_file: str = None):

    devices = []
    if to_screen:
        devices.append(CV2Screen(fps=fps, scale=1.))
    if output_file:
        output_file = pathlib.Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        devices.append(CV2VideoWriter(str(output_file), fps, scale=1.))

    return OutputDeviceList(devices, scale)
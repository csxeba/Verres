import numpy as np

from verres.data import event
from verres.utils import visualize

stream = event.DiffStream("/data/Datasets/Video2Events/Baba.mp4")
screen = visualize.CV2Screen(fps=250)

with stream, screen:
    for frame, dif in stream:
        ndif = (dif / 255.).astype("float32")  # [-1 ... +1]
        pdif = np.abs(ndif)
        pn = TICK_PER_FRAME * pdif
        vdif = np.nan_to_num(ndif / pn, 0)
        canvas = frame / 255.
        for tick in range(TICK_PER_FRAME):
            screen.write((canvas * 255.).astype("uint8"))
            mask = np.random.random(frame.shape) < pdif
            events = vdif[mask]
            locations = np.argwhere(mask)
            canvas[mask] += events
            np.clip(canvas, 0, 1, out=canvas)
            print(f"ACTIVITY LVL: {np.sum(mask) / mask.size:.2%}")

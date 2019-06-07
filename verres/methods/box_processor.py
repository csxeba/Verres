import numpy as np
from skimage.feature import peak_local_max


def process(output, stride):
    heatmaps = output[..., :-4]
    refinements = output[..., -4:-2]
    peaks_found = []
    wh = output[..., -2:]
    for i in range(heatmaps.shape[-1]):
        peaks = peak_local_max(heatmaps[..., i], threshold_abs=0.1, min_distance=2)
        N = len(peaks)
        if N == 0:
            continue
        refine = refinements[tuple(peaks[:, 0]), tuple(peaks[:, 1])]
        peaks_refined = peaks + refine
        peaks_refined *= stride
        peaks_refined = peaks_refined[:, ::-1]

        boxparams = wh[tuple(peaks[:, 0]), tuple(peaks[:, 1])] * stride

        finds = np.empty([N, 5])
        finds[:, :2] = peaks_refined - boxparams
        finds[:, 2:4] = boxparams * 2
        finds[:, -1] = i

        peaks_found.append(finds)
    if len(peaks_found) == 0:
        return np.empty([0, 5])
    peaks_found = np.concatenate(peaks_found)
    areas = peaks_found[:, 2] * peaks_found[:, 3]
    peaks_found = peaks_found[areas > 16]
    return peaks_found

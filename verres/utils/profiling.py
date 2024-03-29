import time
from collections import defaultdict
from typing import List, Dict

import numpy as np


def timed(callback, *args, **kwargs):
    start = time.time()
    result = callback(*args, **kwargs)
    return result, time.time() - start


class Timer:

    def __init__(self):
        self.start = 0.
        self.result = 0.

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result = time.time() - self.start

    def reset(self):
        r = self.result
        self.start = 0.
        self.result = 0.
        return r


class MultiTimer:

    def __init__(self):
        self._results: Dict[List[float]] = defaultdict(list)
        self._currently_measuring = None

    def time(self, fieldname):
        """
        Starts measuring.
        """
        if self._currently_measuring is not None:
            raise RuntimeError("I was already measuring {}")
        self._currently_measuring = fieldname
        self._results[fieldname].append(time.time())
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._results[self._currently_measuring][-1] = time.time() - self._results[self._currently_measuring][-1]
        self._currently_measuring = None

    def get_results(self, reset=False, reduce=True):
        result = self._results
        if reduce:
            result = {k: np.mean(v) for k, v in result.items()}
        if reset:
            self.reset()
        return result

    def reset(self):
        self._results = defaultdict(list)
        self._currently_measuring = None

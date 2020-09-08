import time


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
        self._results = {}
        self._latest = None

    def time(self, fieldname):
        self._latest = fieldname
        self._results[fieldname] = time.time()
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._results[self._latest] = time.time() - self._results[self._latest]

    def get_results(self, field=None, reset=False):
        if field is not None:
            result = self._results[field]
        else:
            result = self._results
        if reset:
            self.reset()
        return result

    def reset(self):
        self._results = {}
        self._latest = None

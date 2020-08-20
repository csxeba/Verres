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

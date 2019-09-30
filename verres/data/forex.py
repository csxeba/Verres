import os
import pathlib

import numpy as np
import h5py

DEFAULT_ROOT = "/data/Datasets/Forex/"
DEFAULT_H5FL = "/data/Datasets/Forex.h5"


def interpolate(x0, y0, x1, y1, x):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))


def parse_csv(path):
    lines = np.array([line.split(",") for line in open(path).read().split("\n")[:-1]])
    dates = np.array(list(map(parse_date, lines[:, 0])))
    bids = lines[:, 1].astype("float64")
    asks = lines[:, 2].astype("float64")
    volumes = lines[:, 3].astype("int64")
    return {"date": dates, "bids": bids, "asks": asks, "vols": volumes}


def parse_date(date):
    date, time = date.split(" ")
    Y, M, D = int(date[:4]), int(date[4:6]), int(date[-2:])
    h, m, s, ms = int(time[:2]), int(time[2:4]), int(time[4:6]), int(time[6:])
    dt = np.datetime64(f"{Y}-{M:0>2}-{D:0>2}T{h:0>2}:{m:0>2}:{s:0>2}.{ms}")
    return dt.astype("float64")


class ForexDataset:

    def __init__(self, h5dataset=DEFAULT_H5FL):
        self.dataset = h5py.File(h5dataset, "r")
        self.lengths = {"eurchf": 10_882_874, "eurhuf": 2_579_841, "eurgbp": 13_296_461, "eurusd": 16_659_128}

    def get_item(self, pair_name, date):
        index = self.dataset[pair_name]["date"].searchsorted(date)
        ask_0 = self.dataset[pair_name]["asks"][index]
        bid_0 = self.dataset[pair_name]["bids"][index]
        vol_0 = self.dataset[pair_name]["vols"][index]
        dat_0 = self.dataset[pair_name]["date"][index]

        ask_1 = self.dataset[pair_name]["asks"][index+1]
        bid_1 = self.dataset[pair_name]["bids"][index+1]
        vol_1 = self.dataset[pair_name]["vols"][index+1]
        dat_1 = self.dataset[pair_name]["date"][index+1]

        ask = interpolate(dat_0, dat_1, ask_0, ask_1, date)
        bid = interpolate(dat_0, dat_1, bid_0, bid_1, date)
        vol = interpolate(dat_0, dat_1, vol_0, vol_1, date)

        return ask, bid, vol

    @classmethod
    def from_files(cls, rootdir=DEFAULT_ROOT, h5file=DEFAULT_H5FL, overwrite=False):
        dataset_root = pathlib.Path(rootdir).absolute()
        if os.path.exists(h5file) and not overwrite:
            raise RuntimeError("H5 file already exists")
        elif os.path.exists(h5file) and overwrite:
            os.remove(h5file)

        shapes = {"eurchf": 10882874, "eurhuf": 2579841, "eurgbp": 13296461, "eurusd": 16659128}

        with h5py.File(h5file) as h5:

            for pair in sorted(dataset_root.glob("*")):
                pair_name = os.path.split(str(pair))[-1]

                g = h5.create_group(pair_name)

                date = g.create_dataset("date", dtype="float64", shape=(shapes[pair_name],))
                bids = g.create_dataset("bids", dtype="float64", shape=(shapes[pair_name],))
                asks = g.create_dataset("asks", dtype="float64", shape=(shapes[pair_name],))
                vols = g.create_dataset("vols", dtype="float64", shape=(shapes[pair_name],))

                pointer_start = 0

                for file in sorted(pair.glob("*")):
                    print("Parsing", file)
                    series = parse_csv(file)
                    pointer_end = pointer_start + len(series["date"])

                    print("Writing to H5...")
                    date[pointer_start:pointer_end] = series["date"]
                    bids[pointer_start:pointer_end] = series["bids"]
                    asks[pointer_start:pointer_end] = series["asks"]
                    vols[pointer_start:pointer_end] = series["vols"]

                    pointer_start = pointer_end

        return cls(h5file)

    def stream(self, batch_size, timestep, infinite=True):
        ...


def main():
    from matplotlib import pyplot as plt
    ds = ForexDataset()
    print("HU")


if __name__ == '__main__':
    main()
AW
import os

import verres as V


def extract_last_epoch(artifactory_root) -> int:
    logfile = os.path.join(artifactory_root, "training_logs.csv")
    if not os.path.exists(logfile):
        print(" [Verres] - Cannot file CSV logs under artifactory root", artifactory_root)
        print(" [Verres] - Training cannot continue, initial_epoch is forced to be 0")
        last_epoch = 0
    else:
        with open(str(logfile)) as f:
            iterator = iter(f)
            header = next(iterator)
            header = header.split(",")
            epoch_idx = header.index("epoch")
            last_epoch = max(int(line.split(",")[epoch_idx]) for line in iterator) + 1
        print(f" [Verres] - Continue train from epoch {last_epoch}")
    return last_epoch

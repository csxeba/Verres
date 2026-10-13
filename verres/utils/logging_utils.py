import os

import verres as V


def extract_last_epoch(artifactory_root) -> int:
    """Epoch to continue from, read from the CSV log of a run (0 if nothing was logged)."""
    logfile = os.path.join(str(artifactory_root), "training_logs.csv")
    if not os.path.exists(logfile):
        print(" [Verres] - Cannot find CSV logs under verres-artifactory root", artifactory_root)
        print(" [Verres] - Training cannot continue, initial_epoch is forced to be 0")
        return 0
    epochs = []
    with open(str(logfile)) as f:
        iterator = iter(f)
        header = next(iterator, "").strip().split(",")
        if "epoch" not in header:
            raise RuntimeError(f"No `epoch` column in the training log: {logfile}")
        epoch_idx = header.index("epoch")
        for line in iterator:
            fields = line.strip().split(",")
            if len(fields) != len(header):
                continue  # a killed run can leave a truncated last row behind
            epochs.append(int(fields[epoch_idx]))
    last_epoch = max(epochs) + 1 if epochs else 0
    print(f" [Verres] - Continue train from epoch {last_epoch}")
    return last_epoch

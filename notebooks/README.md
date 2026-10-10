# Colab training

`colab_train.ipynb` runs one training on a Colab GPU: the repo is cloned, the dataset is downloaded into
the VM, and only the artifactory (checkpoints, logs, KPIs) is kept on Drive.

## 1. Dataset

The notebook cell downloads the official archive itself, nothing has to be uploaded:

    https://www.robots.ox.ac.uk/~vgg/share/cocodoom-v1.0.1.tar.gz   # 2.2 GB

It contains the standard splits `run-{train,val,test}.json` plus `map-*.json` and the RGB frames, and is
extracted into `/content/cocodoom` with `tar -xzf --strip-components=1`. The `run-full-*` annotations and
the 13 GB `cocodoom-full-v1.0.1.tar.gz` are not needed on Colab, so the colab configs use `full: False`.
Download and extraction need ~5 GB of VM disk and are repeated on every new runtime.

## 2. Code

    git push -u origin feature/set_baselines          # then the clone cell works as is
    git archive --format=zip -o /tmp/verres.zip HEAD  # otherwise upload this and set REPO_ZIP, REPO_URL=""

## 3. First run checklist

1. Runtime -> Change runtime type -> T4 GPU; the notebook only warns when no GPU is visible.
2. Run the cells in order and read the `RESUMABLE` list before training.
3. Fresh run: `RUN_DIR = None`. After a disconnect: mount Drive, re-download the dataset, set `RUN_DIR`
   to a `RESUMABLE` path. `EPOCHS` raises `training.epochs` for an already configured run.
4. `artifactory_root` in the colab configs must point at Drive - checkpoints are what survives a disconnect.
5. Evaluation (`execution_type: evaluation`) writes the `*-kpi.json` the KPI cell prints.

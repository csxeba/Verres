# Verres

Deep Learning experimentation framework running on TensorFlow 2.x.

Verres is organized so it can be used either as a framework for running and
managing Deep Learning experiments or a library that provides utilities for
Deep Learning experimentation.

## Installation as a library

If one intends to use Verres as a collection of tools organized as a library,
issuing the following pip command can install it as a package:

```pip3 install git+https://github.com/csxeba/Verres.git ```

## Using as a framework

If one intends to use Verres as an experiment management framework, it is advised
to clone the repository first:

```git clone https://github.com/csxeba/Verres.git```

This provides the user with the following additional tools:
- A **conda** environment yml file
- A **Dockerfile**, that builds an easy-to-use and easy-to-reproduce environment
- A **Makefile**, that orchestrates the docker building process
- A **main** (bash) file, that can quickly Python scripts in the Verres framework. 

## Concepts in Verres

Verres organizes a Deep Learning experiment into the following abstractions:

### Data concepts:

- **Dataset**: holds a dataset and exposes sample IDs (`verres.data.dataset`).
- **Transformation**: turns a sample into network input or ground truth tensors
  (`verres.data.transformation`).
- **Pipeline**: combines a dataset with its transformations. `stream()` and
  `get_tf_dataset()` (`verres.data.streaming`) produce the training stream.

### Model concepts:

The below concepts all represent a derivative of _tf.keras.Model_.

- **Backbone**: task-agnostic, possibly pretrained feature producing part of a neural network.
- **Neck**: optional task-agnostic part, aggregates the backbone features.
- **Head**: task-specific part, generates predictions.

### Input sample (eg. image) preprocessing:
The backbone is responsible for exposing the preprocessing logic, that
preprocesses the network input, so it is compatible with the given backbone.
The backbone does not call this preprocessor, but assumes the inputs given to the
backbone.call() method are treated with the exposed preprocessor.

### Postprocessing
The head is responsible for exposing `call(inputs)`, which describes the forward
pass during inference. Training logic (`train_step`) lives in the
`TrainingExecutor` (`verres.execution`).

## Datasets

The currently supported dataset is available under verres.data.dataset.

- cocodoom: synthetic video dataset with frame-by-frame annotations for objects,
instance masks, depth and a lot more. See https://www.robots.ox.ac.uk/~vgg/research/researchdoom/cocodoom/

## Framework mode

### Docker

A Makefile and a main file are added to ease the execution of arbitrary python
files in the Verres docker environment.

Building and interactively entering the Verres environment can be done by
using our Makefile.

First copy `environment_setup_sample.sh` to `environment_setup.sh` and set the following variables:

- **VERRES_ARTIFACTORY**: where artifacts (logfiles, model checkpoints) go.
  Mounted to `/artifactory`
- **VERRES_MODELS**: where your pretrained weights are.
  Mounted to `/models`
- **VERRES_DATA**: where your datasets are stored (cocodoom for instance)
  Mounted to `/data`
- **VERRES_ACCELERATOR**: either mkl or gpu
- **VERRES_TENSORBOARD_PORT**: where to forward the tensorboard port
- **VERRES_JUPYTER_PORT**: where to forward the jupyter port

The Verres framework root will also be mounted to `/workspace`.

`make` or `make run` will start and attach to an interactive shell.

`make run-jupyter` will start a jupyter server.

`make run-tensorboard` will fire up a TensorBoard server.

An experiment can be executed in the docker environment by issuing

```./main path/to/config.yml [-e EXECUTION_TYPE] [-w MODEL_WEIGHTS] [-u field.path=value ...]```

This runs `python main.py --config ...`. Training can be resumed with
`--continue_training <artifact dir>`.

### Non-docker

If one would not like to use Docker, the environment can also be assembled by

`pip install .`

Or using conda for CPU-based acceleration:

`conda env create --name <your env name> conda/conda-env-mkl.yml`

Or using conda for GPU-based acceleration:

`conda env create --name <your env name> conda/conda-env-gpu.yml`

It is advised to set the PYTHONPATH variable to include the Verres repository
root by

`export PYTHONPATH="$(pwd):${PYTHONPATH}"`

### Examples

Example configs are stored in the `config/` directory, example scripts in `experiments/`.

### Tests and tools

Tests run in a TensorFlow >= 2.16 / Keras 3 environment.

```.venv/bin/python -m pytest tests -q```

`notebooks/colab_train.ipynb` runs a training on a Colab GPU: it clones the repo, downloads the official
COCODoom archive into the VM and keeps the artifactory on Drive (`notebooks/README.md`).

```python experiments/render_video.py config/exp_od_baseline.yml artifactory/od/baseline/xp_... \
    --sequence run2/map01 --frames 0 600 --out out.mp4```

`render_video.py` draws the detections of a trained run on one sequence into a video, and
`inference_time_od.py <config> [--threads N] [--tflite]` benchmarks the forward and forward+decode
latency of a config and prints a JSON summary.

## Status 26Q3

I moved on from TensorFlow to PyTorch. A re-implementation is necessary.

There are good ideas in this repo.

## Thanks!

Thank you for using Verres for your Deep Learning experiments. Issues and
pull requests are always welcome!



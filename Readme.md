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

- **Task**: Specifies the ground truth and output that will be learned (eg. object detection)
- **Data loader**: responsible for holding and transforming a dataset and exposes
an iterator-like interface, that yields (batches of) sample IDs.
- **Data streamer**: wraps the Data loader with the training preprocessing and
ground truth tensor generation logic for training a neural network.

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
The head is responsible for exposing the following methods:
- call(inputs): describes the logic for the forward pass during inference.
- detect(inputs): handles preprocessing, calls *call()* and executes postprocessing.
- train_step(data): handles image preprocessing, executes the forward pass in
gradient tape, calculates gradients and calls optimizer.apply_gradients().

 
## Datasets

Current supported datasets are available under verres.data.

- cocodoom: synthetic video dataset with frame-by-frame annotations for objects,
instance masks, depth and a lot more. See https://www.robots.ox.ac.uk/~vgg/research/researchdoom/cocodoom/
- inmemory: some image classification datasets, that fit into the RAM easily.

## Framework mode

### Docker

A Makefile and a main file are added to ease the execution of arbitrary python
files in the Verres docker environment.

Building and interactively entering the Verres environment can be done by
using our Makefile.

First edit the `.env` file and set the following paths and variables:

- **host_artifactory_root**: where artifacts (logfiles, model checkpoints) go.
  Mounted to `/artifactory`
- **host_models_root**: where your pretrained weights are.
  Mounted to `/models`
- **host_data_root**: where your datasets are stored (cocodoom for instance)
  Mounted to `/data`
- **compute_accelerator**: either mkl or gpu
- **host_tensorboard_port**: where to forward the tensorboard port
- **host_jupyter_port**: where to forward the jupyter port

The Verres framework root will also be mounted to `/workspace`.

`make` or `make run` will start and attach to an interactive shell.

`make jupyter` will start a jupyter server.

`make tensorboard` will fire up a TensorBoard server.

An arbitrary Python script can be executed in the docker environment by issuing

```./main arbitrary_path_to_python_script.py [--arbitrary --params]```

### Non-docker

If one would not like to use Docker, the environment can also be assembled by

`pip install .` or `pip install .[gpu]`

Or using conda for CPU-based acceleration:

`conda env create --name <your env name> conda/conda-env-mkl.yml`

Or using conda for GPU-based acceleration:

`conda env create --name <your env name> conda/conda-env-gpu.yml`

It is advised to set the PYTHONPATH variable to include the Verres repository
root by

`EXPORT PYTHONPATH="${pwd}:${PYTHONPATH}"`

### Examples

Example executions are stored in the `executions/` directory.

## Thanks!

Thank you for using Verres for you Deep Learning experiments. Issues and
pull requests are always welcome!


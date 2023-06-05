# docTR

docTR is a Document Text Recognition system. This version is implemented with [Replicate Cog](https://github.com/replicate/cog). It uses models from the [doctr](https://github.com/mindee/doctr) library to perform text detection and recognition.

## Configuration

`build`: Specifies the build environment for the model, including the Python version, system packages, and Python packages.

`predict`: Specifies the file and function that Cog will use to make predictions. In this case, it is the `predict` function in the `predictor.py` file.

## Setup

To set up the model for use with Cog, a `cog.yaml` file is included in the repository. This file specifies the build and prediction configuration for the model. It also includes the required system packages and Python packages for the model.

The `predictor.py` file contains the `Predictor` class, which is used by Cog to interact with the model. The `Predictor` class includes a `setup` method for loading the model, and a `predict` method for making predictions.

## Running the Model

With Cog installed, you can use the `cog run` command to start a local server and interact with the model:

```shell
$ cog run
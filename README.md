## Train using Amazon SageMaker in script mode and debug using Amazon SageMaker Debugger

This repository contains two examples for performing training on Amazon SageMaker using SageMaker's script mode and debugging using Amazon SageMaker Debugger.

### Overview

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service that provides every developer and data scientist with the ability to build, train and deploy machine learning (ML) models quickly. With SageMaker, you have the option of using the built-in algorithms as well as bringing your own algorithms and frameworks. One such framework is TensorFlow 2.x. Amazon SageMaker Debugger debugs, monitors and profiles training jobs in real time thereby helping with detecting non-converging conditions, optimizing resource utilization by eliminating bottlenecks, improving training time and reducing costs of your machine learning models.

### Example 1: Using default training loop

This example contains a Jupyter Notebook that demonstrates how to use a SageMaker optimized TensorFlow 2.x container to train a model on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) and debug using SageMaker Debugger.  Finally the debugger's output is analyzed.  This will take your training script and use SageMaker in script mode with the default training loop.

#### Repository structure

This repository contains

* [A Jupyter Notebook](https://github.com/aws-samples/amazon-sagemaker-script-mode-with-debugger/blob/main/notebooks/tf2_fashion_mnist_debugger.ipynb) to get started

* [A training script in Python](https://github.com/aws-samples/amazon-sagemaker-script-mode-with-debugger/blob/main/notebooks/scripts/train_tf2_fashion_mnist.py) that is passed to the training job

### Example 2: Using custom training loop

This example contains a Jupyter Notebook that demonstrates how to use a SageMaker optimized TensorFlow 2.x container to train a model on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) and debug using SageMaker Debugger.  Finally the debugger's output is analyzed.  This will take your training script and use SageMaker in script mode with a custom training loop i.e. customizes what goes on in the `fit()` loop.

#### Repository structure

This repository contains

* [A Jupyter Notebook](https://github.com/aws-samples/amazon-sagemaker-script-mode-with-debugger/blob/main/notebooks/tf2_fashion_mnist_custom_debugger.ipynb) to get started

* [A training script in Python](https://github.com/aws-samples/amazon-sagemaker-script-mode-with-debugger/blob/main/notebooks/scripts/train_tf2_fashion_mnist_custom.py) that is passed to the training job

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Distributed training using Amazon SageMaker Distributed Data Parallel library and debugging using Amazon SageMaker Debugger

This repository contains an example for performing distributed training on Amazon SageMaker using SageMaker's Distributed Data Parallel library and debugging using Amazon SageMaker Debugger.

### Overview

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service that provides every developer and data scientist with the ability to build, train and deploy machine learning (ML) models quickly. With SageMaker, you have the option of using the built-in algorithms as well as bringing your own algorithms and frameworks. One such framework is TensorFlow 2.x. When performing distributed training with this framework, you can use SageMaker's Distributed Data Parallel or Distributed Model Parallel libraries. Amazon SageMaker Debugger debugs, monitors and profiles training jobs in real time thereby helping with detecting non-converging conditions, optimizing resource utilization by eliminating bottlenecks, improving training time and reducing costs of your machine learning models.

This example contains a Jupyter Notebook that demonstrates how to use a SageMaker optimized TensorFlow 2.x container to perform distributed training on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using the [SageMaker Distributed Data Parallel library](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html) and debug using [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html). It also implements a custom training loop i.e. customizes what goes on in the fit() loop. Finally the debugger's output is analyzed. This notebook will take your training script and use SageMaker in script mode.

### Repository structure

This repository contains

* [A Jupyter Notebook](https://github.com/aws-samples/amazon-sagemaker-dist-data-parallel-with-debugger/blob/main/notebooks/tf2_fashion_mnist_custom_smd_debugger.ipynb) to get started

* [A training script in Python](https://github.com/aws-samples/amazon-sagemaker-dist-data-parallel-with-debugger/blob/main/notebooks/scripts/train_tf2_fashion_mnist_custom_smd.py) that is passed to the training job

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

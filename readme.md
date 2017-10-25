# Classify MNIST dataset using Chainer

This sample uses [Chainer](https://chainer.org/), the popular flexible framework of neural networks for deep learning from Preferred Networks, to classify the ageless [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwritten digits.

These are minimal Chainer examples to write a feed-forward net.
The code consists of three parts: dataset preparation, network and optimizer definition and learning loop.
This is a common routine to write a learning process of networks with dataset that is small enough to fit into memory.

## Logging code
The sample codes are based on [Chainer Examples](https://github.com/chainer/chainer/tree/master/examples/mnist) on GitHub. The only change we make is to add some Azure ML specific logging code into the experiment (`train_mnist_custom_loop.py`).

Here are the relevant code snippets:
```python
# reference the Azure ML logging library
from azureml.logging import get_azureml_logger
...
# initialize the logger
run_logger = get_azureml_logger()
...
# declare empty lists
metrics = []
losses = []
# during the training session
while train_iter.epoch < args.epoch:
    ...
    # record accuracy and loss into a list
    metrics.append(float(sum_accuracy / test_count))
    losses.append(float(sum_loss))
    ...

# after the training finishes, log the list of accuracies and losses
run_logger.log("Accuracy", metrics)
run_logger.log("Loss", losses)
```

By adding the above logging code, when the run finishes, you can find the following graph plotted for you in the run history detail page.

![cover](./docs/metrics.png)

## Instructions for running scripts from CLI window
You can run the scripts from the Workbench app, but it is more interesting to run it from the command-line window so you can watch the feedback in real-time.

Open the command-line window by clicking on **File** --> **Open Command Prompt**, then run `train_mnist_custom_loop.py` in local Python environment installed by Azure ML Workbench by typing in the following command.
```
# first install chainer library using pip, you only needed to do this once.
$ pip install chainer

# submit the experiment to local execution environment
$ az ml experiment submit -c local train_mnist_custom_loop.py
```

If you have Docker engine running locally, you can run `train_mnist_custom_loop.py` in a Docker container.

>Note: this command automatically pulls down a base Docker image so it can take a few minutes before the job is started. But this only happens if you are running it for the first time. The subsequent runs will be much faster.

And you don't need to pip-install the _chainer_ library, since it is already specified in the `conda_dependencies.yml` file under `aml_config` folder. The execution engine will automatically install it as part of the Docker image building process.

```
# submit the experiment to local Docker container for execution
$ az ml experiment submit -c docker train_mnist_custom_loop.py
```

You can also run `train_mnist_custom_loop.py` in a Docker container in a remote machine. Note you need to create/configure myvm.compute.
```
# attach a new compute context
$ az ml computetarget attach --name myvm --address <ip address or FQDN> --username <username> --password <pwd> --type remotedocker

# prepare the Docker image
$ az ml experiment prepare -c myvm

$ az ml experiment submit -c myvm train_mnist_custom_loop.py
```

## Tiding a trained model and output images into your run history
You can write files to a folder named _outputs_ that's relative to the root directory. The folder receives special treatment by the Experimentation service. Anything your script creates in the folder during the execution, such as a model file, data file, or plotted image file (collectively known as artifacts), is copied to the Azure Blob storage account that's associated with your experimentation account after the run is finished. The files become part of your run history record.

If you want to check how this works, use `train_mnist.py`. Output images are shown in the run history detail page like below.

![visualizations](./docs/visualizations.png)

The default output directory is set to _outputs_ in `train_mnist.py` and `train_mnist_custom_loop.py` by this. 

```python
parser.add_argument(
    '--out',
    '-o',
    default='outputs',
    help='Directory to output the result')
```

## Running it on a VM with GPU
With computationally expensive tasks like training a neural network, you can get a huge performance boost by running it on a GPU-equipped machine.


>Note, if your local machine already has NVidia GPU chips, and you have installed the CUDA libraries and toolkits, you can directly run the script using local compute target. Just be sure to pip-install the _cupy_ Python package. The below instructions are specifically for running script in a remote VM equipped with GPU.

>Azure Machine Learning Services is currently in preview and can't install cupy in docker image because the base docker image lacks of some components. This will be fixed in the future.

### Step 1. Provision a GPU Linux VM 
Create an Ubuntu-based Data Science Virtual Machine(DSVM) in Azure portal using one of the NC-series VM templates. NC-series VMs are the VMs equipped with GPUs for computation.

### Step 2. Attach the compute context
Run following command to add the GPU VM as a compute target in your current project:
```
$ az ml computetarget attach --name myvm --address <ip address or FQDN> --username <username> --password <pwd> --type remotedocker
```
The above command creates a `myvm.compute` and `myvm.runconfig` file under the `aml_config` folder.

### Step 3. Modify the configuration files under _aml_config_ folder
- You need the cupy library built for GPU:
    
    In `conda_dependencies.yml` file, add `cupy`.

- You need a different base Docker image with CUDA libraries preinstalled:

    In `myvm.compute` file, replace the value of `baseImage` from `microsoft/mmlspark:plus-0.7.91` to  `microsoft/mmlspark:plus-gpu-0.7.91`

- You need to use _NvidiaDocker_ command to start the Docker container as opposed to the regular _docker_ command.

    In `myvm.compute` file, add a line: `nvidiaDocker: true`

- You need to specify the run time framework as _Python_ as opposed to _PySpark_:

    In `myvm.runconfig` file,  change the value of `Framework` from `PySpark` to `Python`.

### Step 4. Run the script.
Now you are ready to run the script.
```
$ az ml experiment submit -c myvm train_mnist_custom_loop.py
```
You should notice the script finishes significantly faster than than if you use CPU. And the command-line outputs should indicate that GPU is used for executing this script.

If you want to run this example on the N-th GPU, pass `--gpu=N` to the script.
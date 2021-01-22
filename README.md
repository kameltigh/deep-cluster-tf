# deep-cluster-tf

An implementation of the Facebook Research [DeepCluster](https://github.com/facebookresearch/deepcluster) algorithm
using Tensorflow and applied to medical images of lungs.

### Installation

1. Download or git clone the current repository.
2. Install _Poetry_ (follow the [official website instructions](https://python-poetry.org/docs/#installation)).
3. Create the project environment and install its dependencies by running: `make install`.
4. Download the RSNA lung images dataset from [here](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
   and put it in the data folder.

### Usage

Simply run `make run` after having followed the installation process.

### Non-centered images

Some images of the dataset are not centered and contain a big proportion of zero pixels. Since those are a minority, we
simply remove them using a python script. You can run it using `make remove_non_centered_image`
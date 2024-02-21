# Seeded Poisson Factorization

Source code for the paper: *link*

## Installation for GPU

Configure a virtual environment using Pyhton 3.9+. Inside the virtual environment, use `pip` to install the required packages:

```{bash}
(venv)$ pip install -r requirements.txt
```

The main dependencies are Tensorflow (1.10.0) and Tensorflow Probability (0.14.0)


## Data

We are using customer feedback from the Amazon dataset, available [here](https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification). The dataset in this repo was already preprocessed. However, SPF can be applied to any corpus of text data.


## Applying SPF

To apply SPF see [tbip.py](https://github.com/keyonvafa/tbip/blob/master/tbip.py) to produce ideal points.


## Reproducing Paper Results

To reproduce our paper results please see [reproduction section](https://github.com/keyonvafa/tbip/blob/master/tbip.py) and run the respective files.
# Seeded Poisson Factorization

Source code for the paper: [Seeded Poisson Factorization: Leveraging domain knowledge to fit topic models](https://arxiv.org/abs/2503.02741).

## Installation

Configure a virtual environment using Pyhton 3.10+. Inside the virtual environment, use `pip` to install the required packages:

```{bash}
(venv)$ pip install -r requirements.txt
```

The main dependencies are Tensorflow (2.15.0) and Tensorflow Probability (0.23.0). Be sure to adjust the dependencies if you are able to accelerate GPU support.


## Data

We are using customer feedback from the Amazon dataset, available [here](https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification). The dataset in this repo was already preprocessed. However, SPF can be applied to any corpus of text data.


## Applying SPF

To apply the SPF model, see the [example notebook](minimal_example.ipynb) for a minimal example.


## Reproducing Paper Results

To reproduce our paper results please see [reproduction section](https://github.com/BPro2410/Seeded-Poisson-Factorization/tree/main/analysis/reproduction).
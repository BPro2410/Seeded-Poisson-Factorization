# Seeded Poisson Factorization

Source code for the paper: [Seeded Poisson Factorization: leveraging domain knowledge to fit topic models](https://www.sciencedirect.com/science/article/pii/S095070512501161X).

This repo contains an easy to use implementation of the Seeded Poisson Factorization (SPF) topic model. SPF is a guided topic modeling approach that allows users to pre-specify topics of interest by providing sets of seed words. Built on Poisson factorization, it leverages variational inference techniques for efficient and scalable computation. 

<p>
    <div align="center">
        <img src="./seededpf/spf_graphical.PNG" width="50%" alt/>
    </div>
</p>



## Installation

The model works with **Python 3.10** or **Python 3.11**. The main dependencies are Tensorflow 2.18 and tensorflow_probability 0.25. 

> Please be sure to _adjust the dependencies if you are able to accelerate GPU support_.

### Via pip

SPF is available on [PyPI](https://pypi.org/project/seededPF/). The easiest way to install the model is via `pip`.

```{bash}
pip install seededpf
```

Afterwards, the SPF model can be imported:
```python
from seededpf import SPF
```


### From source
Configure a virtual environment using Pyhton 3.10 or Python 3.11. Inside the virtual environment, use `pip` to install the required packages:

```{bash}
(venv)$ pip install -r requirements.txt
```

## Data

We are using customer feedback from the Amazon dataset, available [here](https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification). The dataset in this repo was already preprocessed. However, SPF can be applied to any corpus of text data.


## Applying SPF

To apply the SPF model, see [a minimal example](minimal_example.ipynb) or [an advanced example](analysis/examples/SPF_example_notebook.ipynb).


## Reproducing Paper Results

To reproduce our paper results please see [reproduction section](https://github.com/BPro2410/Seeded-Poisson-Factorization/tree/main/analysis/reproduction).


## Citing

When citing `seededPF`, please use this BibTeX entry:

```
@article{PROSTMAIER2025114116,
    title = {Seeded Poisson Factorization: leveraging domain knowledge to fit topic models},
    journal = {Knowledge-Based Systems},
    volume = {327},
    pages = {114116},
    year = {2025},
    issn = {0950-7051},
    doi = {https://doi.org/10.1016/j.knosys.2025.114116},
    url = {https://www.sciencedirect.com/science/article/pii/S095070512501161X},
    author = {Bernd Prostmaier and Jan Vávra and Bettina Grün and Paul Hofmarcher},
    keywords = {Poisson factorization, Topic model, Variational inference, Customer feedback},
    abstract = {Topic models are widely used for discovering latent thematic structures in large text corpora, yet traditional unsupervised methods often struggle to align with pre-defined conceptual domains. This paper introduces seeded Poisson factorization (SPF), a novel approach that extends the Poisson factorization (PF) framework by incorporating domain knowledge through seed words. SPF enables a structured topic discovery by modifying the prior distribution of topic-specific term intensities, assigning higher initial rates to pre-defined seed words. The model is estimated using variational inference with stochastic gradient optimization, ensuring scalability to large datasets. We present in detail the results of applying SPF to an Amazon customer feedback dataset, leveraging pre-defined product categories as guiding structures. SPF achieves superior performance compared to alternative guided probabilistic topic models in terms of computational efficiency and classification performance. Robustness checks highlight SPF’s ability to adaptively balance domain knowledge and data-driven topic discovery, even in case of imperfect seed word selection. Further applications of SPF to four additional benchmark datasets, where the corpus varies in size and the number of topics differs, demonstrate its general superior classification performance compared to the unseeded PF model.}
}
```

## License

Code licensed under [MIT](https://github.com/BPro2410/Seeded-Poisson-Factorization/blob/main/LICENSE).


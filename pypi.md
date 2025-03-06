<h1 align="center">SeededPF</h1>

<div align="center">
    <a href="https://pypi.org/project/seededpf">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/seededpf?color=blue">
    </a>
    <a href="https://www.python.org/downloads/">
        <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/seededpf">
    </a>
    <a href="https://github.com/machine-intelligence-laboratory/seededpf/blob/master/LICENSE.txt">
        <img alt="License" src="https://img.shields.io/pypi/l/seededpf?color=Black">
    </a>
</div>

## What is seededPF
`seededPF` is an easy to use implementation of the Seeded Poisson Factorization (SPF) topic model, shown in [this research paper](https://arxiv.org/abs/2503.02741). SPF is a guided topic modeling approach that allows users to pre-specify topics of interest by providing sets of seed words. Built on Poisson factorization, it leverages variational inference techniques for efficient and scalable computation. 

<p>
    <div align="center">
        <img src="./seededpf/spf_graphical.PNG" width="50%" alt/>
    </div>
</p>


Traditional unsupervised topic models often struggle to align with predefined conceptual domains and typically require significant post-processing efforts, such as topic merging or manual labeling, to ensure topic coherence. `seededPF` overcomes this limitation by enabling the pre-specification of topics, which leads to improved topic interpretability and reduces the need for manual post-processing. Additionally, it supports the estimation of unsupervised topics when no seed words are provided.

Consider using `seededPF`  if:
- You need to fit a topic model with a specific topic schema.
- You wish to estimate a topic model that is partially or fully unsupervised (i.e., providing no seed words means fitting a standard Poisson factorization topic model without predefined topics).
- You require a fast and scalable topic modeling solution.

`seededPF` offers a high-performance, scalable interface for topic modeling, providing a reliable alternative to [keyATM](https://keyatm.github.io/keyATM/index.html) and [SeededLDA](https://github.com/koheiw/seededlda), while minimizing the need for manual intervention and enhancing topic interpretability.


## Installation


`seededPF` works with **Python 3.10** or **Python 3.11**. The main dependencies are Tensorflow 2.18 and tensorflow_probability 0.25. 

> Please be sure to _adjust the dependencies if you are able to accelerate GPU support_.

### Via pip

The easiest way to install `seededPF` is via `pip`.

```{bash}
pip install seededpf
```

### From source

One can also install the package from [GitHub](https://github.com/BPro2410/Seeded-Poisson-Factorization). Configure a virtual environment using Pyhton 3.10 or Python 3.11. Inside the virtual environment, use `pip` to install the required packages:

```{bash}
(venv)$ pip install -r requirements.txt
```


# Training the Seeded Poisson Factorization topic model

`seededPF` is an easy to use library for topic modeling. We quickly walk through the most essential steps below:
1. Imports and data preparation
2. Initialization
3. Reading documents
4. Training the model
5. Post-hoc analysis

The following minimal example is available on [GitHub](https://github.com/BPro2410/Seeded-Poisson-Factorization/blob/main/minimal_example.ipynb).

## Step 1: Imports and data preparation

Once installed, one can import the `SPF` class of the `seededPF` library and is ready to go. There are only 2 things required to fit the SPF topic model:
1. Text documents
2. A seed word (i.e. keyword) dictionary for each topic to be estimated.

```python
# Imports
from seededpf import SPF
from sklearn.feature_extraction.text import CountVectorizer

# Example documents - customer reviews about either smartphones or computers
documents = [
    "My smartphone's battery life is fantastic, lasts all day!",
    "The camera on my phone is incredible, takes crystal-clear photos.",
    "Love the smooth performance, but it overheats with heavy apps.",
    "This phone charges super fast, very convenient.",
    "Upgraded my PC and it boots in seconds!",
    "Great for gaming, but gets hot after long sessions.",
    "My computer sometimes freezes, but a restart fixes it.",
    "Best laptop I’ve owned, powerful and reliable!"
]

# Define topic-specific seed words
smartphone = {"smartphone", "iphone", "phone", "touch", "app"}
pc = {"laptop", "keyboard", "desktop", "pc"}

keywords = {"smartphone": smartphone, "pc": pc}
```

## Step 2: Initialization

Now that we have both the documents and the pre-specification of topics to be estimated, we can initialize the SPF topic model.

```python
spf = SPF(keywords = keywords, residual_topics = 0) # Fits 2 seeded topics and 0 unsupervised topics
```

## Step 3: Reading documents

We tokenize the documents and create all data required for model training automatically.

```python
spf.read_docs(documents, 
            count_vectorizer=CountVectorizer(stop_words="english", min_df = 0), 
            batch_size = 1024)
```

## Step 4: Training the model
For model training, we have to set the learning rate and the number of epochs.

```python
spf.model_train(lr = 0.1, epochs = 150)
```


## Step 5: Analysis of the results

There are different methods available to analyze the topic model results. We refer to the [minimal example](https://github.com/BPro2410/Seeded-Poisson-Factorization/blob/main/minimal_example.ipynb) or [advanced example](https://github.com/BPro2410/Seeded-Poisson-Factorization/blob/main/analysis/examples/SPF_example_notebook.ipynb) where we show post-hoc analysis methods.


The `seededPF` package offers several methods, including:
- `SPF.plot_model_loss()`: Checks convergence of the negative ELBO.
- `SPF.return_topics()`: Returns a tuple (categories, E_theta), with categories being the most probable topic for each document and E_theta being the approximate posterior mean estimates per document and topic.
- `SPF.calculate_topic_word_distributions()`: Returns a pandas dataframe containing the approximate topic-term mean intensities.
- `SPF.print_topics()`: Returns a dictionary with the highest intensity words per topic.
- `SPF.plot_seeded_topic_distribution()`: Plots the variational topic word distribution of all seed words belonging to the topic parameter.
- `SPF.plot_word_distribution()`: Shows the fitted variational distribution of q(\Tilde{\beta}){topic,word} and q(\beta^*)_{topic,word}.


# Contribution

If you encounter any bugs or would like to suggest new features for the library, please feel free to contact us or create an issue.

# Citing

When citing `seededPF`, please use this BibTeX entry:

```
@misc{prostmaier2025seededpoissonfactorizationleveraging,
      title={Seeded Poisson Factorization: Leveraging domain knowledge to fit topic models}, 
      author={Bernd Prostmaier and Jan Vávra and Bettina Grün and Paul Hofmarcher},
      year={2025},
      eprint={2503.02741},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2503.02741}, 
}
```

# License

Code licensed under [MIT](https://github.com/BPro2410/Seeded-Poisson-Factorization/blob/main/LICENSE).
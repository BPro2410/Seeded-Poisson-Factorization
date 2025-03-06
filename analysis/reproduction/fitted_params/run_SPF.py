# Imports
import numpy as np
import pandas as pd
import tensorflow as tf

# Set seed
tf.random.set_seed(42)

# Import seededPF
from seededPF.SPF_model import SPF

###################
## Preliminaries ##
###################

# Define keywords
pets = ["dog","cat", "litter", "cats", "dogs", "food", "box", "collar", "water", "pet"]
toys = ["toy", "game", "play", "fun", "old", "son", "year", "loves", "kids", "daughter"]
beauty = ["hair", "skin", "product", "color", "scent", "smell", "used", "dry", "using", "products"]
baby = ["baby", "seat", "diaper", "diapers", "stroller", "bottles", "son", "pump", "gate", "months"]
health = ["product", "like", "razor", "shave", "time", "day", "shaver", "better", "work", "years"]
grocery = ["tea", "taste", "flavor", "coffee", "sauce", "chocolate", "sugar", "eat", "sweet", "delicious"]

keywords = {"pet supplies": pets, "toys games": toys, "beauty": beauty, "baby products": baby, "health personal care": health, "grocery gourmet food": grocery}

# Load data
df1 = pd.read_csv("./data/30k_amazon.csv")

#########
## seededPF ##
#########

# -- Initialize the model
spf1 = SPF(keywords = keywords, residual_topics = 0)
spf1

# -- Read documents and create the data required in the backend
spf1.read_docs(df1["Text"])

# -- Train the model
spf1.model_train(lr = 0.1, epochs = 150, tensorboard = False, early_stopping = False, print_information=True)

# -- Analyze model results
categories, E_theta = spf1.return_topics()
betas = spf1.calculate_topic_word_distributions()
spf1.model_metrics

# -- Calculate model accuracy
df1["SPF_estimates"] = categories
df1["Accuracy"] = df1.Cat1 == df1.SPF_estimates
np.sum(df1.Accuracy) / df1.shape[0]

from sklearn.metrics import classification_report
import pprint
pprint.pprint(classification_report(df1.Cat1, df1.SPF_estimates))

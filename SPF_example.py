# Imports
import numpy as np
import pandas as pd

from SPF.SPF_model import SPF

###################
## Preliminaries ##
###################

# Define keywords
pets = ["dog", "cat", "cats", "dogs", "box", "food", "toy", "pet"]
toys = ["toy", "game", "play", "fun", "son", "daughter", "kids", "playing", "christmas", "toys", "child", "gift"]
beauty = ["skin", "color", "scent", "smell", "dry", "face", "look", "fragrance", "products", "dry", "perfume", "shampoo"]
baby = ["baby", "seat", "son", "daughter", "newborn", "months", "diaper", "diapers", "car", "stroller", "pump", "bag", "child"]
health = ["try", "long", "water", "feel", "shave", "razor", "shaver", "pain", "heart", "tooth", "balm", "pads", "taste", "drink","medical", "blade", "oil", "plastic", "shaker", "soap"]
grocery = ["tea", "taste", "flavor", "coffee", "chocolate", "sugar", "milk", "delicious", "water", "diet"]

keywords = {"pet supplies": pets, "toys games": toys, "beauty": beauty, "baby products": baby, "health personal care": health, "grocery gourmet food": grocery}

# Load data
df1 = pd.read_csv("./data/10k_amazon.csv")


#########
## SPF ##
#########

# Initialize the model
spf1 = SPF(keywords = keywords, residual_topics=0)
spf1

# Read documents and create the data required in the backend
spf1.read_docs(df1["Text"])
# spf1.__dict__

# Train the model
spf1.model_train(lr = 0.1, epochs = 150, tensorboard = False)
# spf1.model_train(lr = 0.1, epochs = 150, tensorboard = True, log_dir = "C:/Users/Bernd/Downloads/test")
# access via cmd: tensorboard --logdir=C:/Users/Bernd/Downloads/test

# See model results
spf1.plot_model_loss(neg_elbo = True)
categories, E_theta = spf1.calculate_topics()
betas = spf1.calculate_topic_word_distributions()

# Calculate model accuracy
df1["SPF_estimates"] = categories
df1["Accuracy"] = df1.Cat1 == df1.SPF_estimates
np.sum(df1.Accuracy) / df1.shape[0]

from sklearn.metrics import classification_report
import pprint
pprint.pprint(classification_report(df1.Cat1, df1.SPF_estimates))



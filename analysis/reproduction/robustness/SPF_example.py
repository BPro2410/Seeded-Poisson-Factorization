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


#########################################
#########################################
######      ROBUSTNESS CHECKS     #######
#########################################
#########################################

# Varying the following variables for Table 5 in section 4.3 of the paper
keywords_short = {k: v[:5] for k,v in keywords.items()}
BATCHSIZE = 1024
LR = 0.01
EPOCHS = 150
UNSEEDED_TOPICS = 0
DOCUMENTS = "30k_amazon.csv"
BETA_TILDE_SHAPE = 1.0



#########
## seededPF ##
#########

# Load data
df1 = pd.read_csv("./data/" + DOCUMENTS)
keywords = keywords
# keywords = keywords_short

# -- Initialize the model
spf1 = SPF(keywords = keywords, residual_topics = UNSEEDED_TOPICS)
spf1

# -- Read documents and create the data required in the backend
spf1.read_docs(df1["Text"], batch_size = BATCHSIZE)

# -- Train the model
spf1.model_train(lr = LR,
                 epochs = EPOCHS,
                 print_information=True,
                 priors = {"beta_tilde_shape": BETA_TILDE_SHAPE})

# -- Analyze model results
categories, E_theta = spf1.return_topics()

# -- Calculate model accuracy
df1["SPF_estimates"] = categories
df1["Accuracy"] = df1.Cat1 == df1.SPF_estimates

from sklearn.metrics import classification_report
import pprint
pprint.pprint(classification_report(df1.Cat1, df1.SPF_estimates))

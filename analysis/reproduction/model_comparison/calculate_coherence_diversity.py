# Imports
import numpy as np
import pandas as pd
import tensorflow as tf

# Set seed
tf.random.set_seed(42)

# Import seededpf
from seededpf.SPF_model import SPF

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


##############
## seededpf ##
##############

# -- Initialize the model
spf1 = SPF(keywords = keywords, residual_topics = 0)
spf1

# -- Read documents and create the data required in the backend
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", min_df = 2)
spf1.read_docs(df1["Text"], count_vectorizer=cv,batch_size = 1024)

# -- Train the model
spf1.model_train(lr = 0.1, epochs = 150, tensorboard = False, early_stopping = False, print_information=True)

# -- Calculate model accuracy
from sklearn.metrics import classification_report
import pprint
pprint.pprint(classification_report(df1.Cat1, spf1.return_topics()[0]))

# -- compute coherence --
top_words = spf1.print_topics(num_words = 10)
top_words_list = list(top_words.values())

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


def compute_coherence(top_words, texts, coherence='c_v'):
    """
    Compute topic coherence using Gensim.

    Parameters
    ----------
    top_words : list of list of str
        Top-N words per topic.
    texts : list of list of str
        Tokenized corpus used for training.
    coherence : str
        One of 'u_mass', 'c_v', 'c_uci', 'c_npmi'
    """
    dictionary = Dictionary(texts)
    cm = CoherenceModel(
        topics=top_words,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence
    )
    return cm.get_coherence()

def compute_topic_diversity(top_words):
    """
    Compute topic diversity: fraction of unique words in top-N lists.
    """
    unique_words = set(word for topic in top_words for word in topic)
    total_words = len(top_words) * len(top_words[0])
    return len(unique_words) / total_words


# get tokenized texts for coherence
vocab = cv.get_feature_names_out()
vocab_lookup = {i: word for word, i in cv.vocabulary_.items()}

def get_tokenized_texts(X, vocab_lookup):
    tokenized_texts = []
    for doc_idx in range(X.shape[0]):
        row = X[doc_idx]
        word_indices = row.indices
        tokenized_texts.append([vocab_lookup[i] for i in word_indices])
    return tokenized_texts

texts = get_tokenized_texts(cv.fit_transform(df1.Text), vocab_lookup)

evaluate_quality_metrics = True
if evaluate_quality_metrics:
    coherence_npmi = compute_coherence(top_words_list, texts, coherence='c_npmi')
    coherence_umass = compute_coherence(top_words_list, texts, coherence='u_mass')
    coherence_cv = compute_coherence(top_words_list, texts, coherence='c_v')  # or 'u_mass'
    diversity = compute_topic_diversity(top_words_list)

    print(f"NPMI: {coherence_npmi:.2f}\nUMASS:{coherence_umass:.2f}\n\CV:{coherence_cv:.2f}\nDiversity:{diversity:.2f}")


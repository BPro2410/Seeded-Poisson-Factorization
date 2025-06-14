## -- Load 20 Newsgroup --
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from seededpf.SPF_model import SPF
from seededpf.utils import clean

# Set seed
tf.random.set_seed(42)

# - load ng
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
newsgroups.keys()

df1 = pd.DataFrame({
    "text": newsgroups.data,
    "label_id": newsgroups.target,
    "label_name": [newsgroups.target_names[i] for i in newsgroups.target]
})

# clean text
df1.text = df1.text.apply(clean)

# - Keywords -
path = "./analysis/reproduction/experiments/Newsgroups/ng2_KW_tfidf.json"
with open(path, "r") as f:
    keywords = json.load(f)

keywords = {k: v[:5] for k, v in keywords.items()}


# remove documents with 0 tokens after tokenization
X = CountVectorizer(stop_words="english", max_features=25000).fit_transform(df1.text)
non_zero_tokens = X.getnnz(axis=1)
df1 = df1[non_zero_tokens > 0]

# - Run SPF -
# -- Initialize the model
spf1 = SPF(keywords=keywords, residual_topics=0)
spf1

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english", max_features=25000)
spf1.read_docs(df1.text, count_vectorizer=cv, batch_size=1024)

# -- Train the model
spf1.model_train(lr=0.01, epochs=550, tensorboard=False, early_stopping=False, print_information=True)

loss_fig, loss_ax = spf1.plot_model_loss(save_path=None)
loss_fig.show()

from sklearn.metrics import classification_report
import pprint

pprint.pprint(classification_report(df1.label_name, spf1.return_topics()[0]))

# coherence
top_words = spf1.print_topics(num_words=10)
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


texts = get_tokenized_texts(cv.fit_transform(df1.text), vocab_lookup)

evaluate_quality_metrics = True
if evaluate_quality_metrics:
    coherence_npmi = compute_coherence(top_words_list, texts, coherence='c_npmi')
    coherence_umass = compute_coherence(top_words_list, texts, coherence='u_mass')
    coherence_cv = compute_coherence(top_words_list, texts, coherence='c_v')  # or 'u_mass'
    diversity = compute_topic_diversity(top_words_list)

    print(f"NPMI: {coherence_npmi:.2f}\nUMASS:{coherence_umass:.2f}\n\CV:{coherence_cv:.2f}\nDiversity:{diversity:.2f}")

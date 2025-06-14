## -- Load 20 Newsgroup --
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

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

# remove documents with 0 tokens after tokenization
X = CountVectorizer(stop_words="english", max_features=25000).fit_transform(df1.text)
non_zero_tokens = X.getnnz(axis=1)
df1 = df1[non_zero_tokens > 0]

# Imports
import numpy as np
import tensorflow as tf

# Set seed
tf.random.set_seed(42)

# Import seededpf
from seededpf.SPF_model import SPF

# -- No domain information --
keywords = dict()

# - Run SPF -
# -- Initialize the model
spf1 = SPF(keywords=keywords, residual_topics=20)
spf1

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words="english", max_features=25000)
spf1.read_docs(df1.text, count_vectorizer=cv, batch_size=1024)

# -- Train the model
spf1.model_train(lr=0.01, epochs=550, tensorboard=False, early_stopping=False, print_information=True)

# -- Analyze model results
loss_fig, loss_ax = spf1.plot_model_loss(save_path=None)
loss_fig.show()

# -- assign classes
E_theta = spf1.variational_parameter["a_theta_S"] / spf1.variational_parameter["b_theta_S"]
categories = np.argmax(E_theta, axis=1)
df1["predicted"] = categories

# EVALUATE/MAXIMIZE CLASSIFICATION PERFORMANCE
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Step 1: Get unique true labels and ensure there are exactly K categories
true_labels = df1['label_name'].values
unique_categories = np.unique(true_labels)
assert len(unique_categories) == 20, "Number of unique categories must be 6 to match predictions 0-19"

# Step 2: Create a mapping of true labels to indices for the confusion matrix
label_to_index = {label: idx for idx, label in enumerate(unique_categories)}

# Convert true labels to indices
y_true_indices = np.array([label_to_index[label] for label in true_labels])

# Step 3: Compute the confusion matrix
# Rows: true labels (indexed 0 to 5), Columns: predicted labels (0 to 5)
cm = confusion_matrix(y_true_indices, categories)

# Step 4: Use the Hungarian algorithm to maximize the sum of correct predictions
# Since linear_sum_assignment minimizes cost, negate the confusion matrix
row_indices, col_indices = linear_sum_assignment(-cm)

# Step 5: Create the mapping from predicted labels to true labels
pred_to_true = {col: unique_categories[row] for row, col in zip(row_indices, col_indices)}

# Step 6: Map predictions to category names
y_pred_mapped = np.array([pred_to_true[pred] for pred in categories])

# Step 7: Calculate accuracy
accuracy = np.mean(y_pred_mapped == true_labels)
print(f"Optimal Accuracy: {accuracy:.4f}")

# Print the mapping
print("Predicted Label to True Label Mapping:")
for pred, true in pred_to_true.items():
    print(f"Predicted {pred} -> True {true}")

# Optional: Evaluate performance with additional metrics
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(true_labels, y_pred_mapped))

# -- compute coherence --
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

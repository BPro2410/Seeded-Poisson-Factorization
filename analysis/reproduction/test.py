
#####################################################
#### Define Keywords + Beta Tilde Hyperparameter ####
#####################################################

# Tests zum Verbessern:
# - reconstruction loss durch datengröße skalieren
# - batch size anpassen

#########################
### batch reshape theta #
#########################

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tf.random.set_seed(2410)
np.set_printoptions(suppress=True)
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sparse

# Shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfp.distributions.JointDistributionCoroutine.Root


import warnings
warnings.filterwarnings("ignore")


pets = ["dog", "cat", "cats", "dogs", "box", "food", "toy", "pet"]
toys = ["toy", "game", "play", "fun", "son", "daughter", "kids", "playing", "christmas", "toys", "child", "gift"]
beauty = ["skin", "color", "scent", "smell", "dry", "face", "look", "fragrance", "products", "dry", "perfume", "shampoo"]
baby = ["baby", "seat", "son", "daughter", "newborn", "months", "diaper", "diapers", "car", "stroller", "pump", "bag", "child"]
health = ["try", "long", "water", "feel", "shave", "razor", "shaver", "pain", "heart", "tooth", "balm", "pads", "taste", "drink","medical", "blade", "oil", "plastic", "shaker", "soap"]
grocery = ["tea", "taste", "flavor", "coffee", "chocolate", "sugar", "milk", "delicious", "water", "diet"]


# TOP 10 Keywords nach TF-IDF
pets = ["dog","cat", "litter", "cats", "dogs", "food", "box", "collar", "water", "pet"]
toys = ["toy", "game", "play", "fun", "old", "son", "year", "loves", "kids", "daughter"]
beauty = ["hair", "skin", "product", "color", "scent", "smell", "used", "dry", "using", "products"]
baby = ["baby", "seat", "diaper", "diapers", "stroller", "bottles", "son", "pump", "gate", "months"]
health = ["product", "like", "razor", "shave", "time", "day", "shaver", "better", "work", "years"]
grocery = ["tea", "taste", "flavor", "coffee", "sauce", "chocolate", "sugar", "eat", "sweet", "delicious"]


keywords = {"pets": pets, "toys": toys, "beauty": beauty, "baby": baby, "health": health, "grocery": grocery}

df1 = pd.read_csv(f"./keyATM/paper_code/10k_amazon.csv")
# df1 = df1.sample(25000, replace = False)
# df1 = df1.sample(500000, replace = True)
variant = "30k"
early_stopping = False
save_files = False

# EPOCH RUNTIMES
# 100k  : 2.15
# 200k  : 5.07
# 300k  : 8.23
# 400k  : 12.5
# 500k  : 17.25
# 600k  : 23.07
# 700k  : 28.48
# 800k  : 35.07
# 900k  : 43.35
# 1000k : 50.167


num_docs = df1.shape[0]
model_hyperparameter = {
    "theta_shape": 0.3, "theta_rate": .3,
    "beta_shape": 0.3, "beta_rate": .3,
    "beta_tilde_shape": 5.0, "beta_tilde_rate": 0.3,
    "learning_rate": 0.1, "num_epochs": 150, "batch_size": 1024,
    "lr_scheduler" : "dynamic",
    "t_S_S": 1., "t_R_S": num_docs/1000,
    "b_S_S": 1., "b_R_S": num_docs/1000*2
}

# Alternative dict Schreibweise!
model_hyper2 = dict(
    theta_shape = .3, theta_rate = .3
)

# def count_words(text):
#     return len(text.split())
#
# df1["tokens"] = df1.Text.apply(count_words)
#
# df1 = df1[(df1.tokens >= 35)]


cv = CountVectorizer(stop_words='english', min_df = 2)


cv.fit(df1["Text"])
vocab = cv.get_feature_names_out()
len(vocab)


num_words = len(vocab)
num_documents = df1.shape[0]
#### Create TBIP Input
counts = sparse.csr_matrix(cv.transform(df1["Text"]), dtype = np.float32)


shuffle = True
if shuffle == True:
    # Shuffle data
    random_state = np.random.RandomState(2410)
    documents = random_state.permutation(num_documents) # shufflet die zahlen von 1:num_documents
    shuffled_counts = counts[documents] # geshuffelte counts dtm nach neuen indices anhand der 'documents'-shuffles
    count_values = shuffled_counts.data # non-zero Einträge der dtm

    shuffled_counts = tf.SparseTensor(
        indices = np.array(shuffled_counts.nonzero()).T,
        values = count_values,
        dense_shape = shuffled_counts.shape
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"document_indices":documents},
         shuffled_counts)
    )
    batch_size = model_hyperparameter["batch_size"]
    ##original
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(1)
    # .prefetch(1) S. 450 Aurelion Textbook - while training on batch 1, the dataset will be working on getting the
    # next batch ready

if shuffle == False:
    count_values = counts.data
    counts = tf.SparseTensor(
        indices = np.array(counts.nonzero()).T,
        values = count_values,
        dense_shape = counts.shape
    )
    documents = np.array(range(num_documents))
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"document_indices":documents},
         counts)
    )
    batch_size = model_hyperparameter["batch_size"]
    dataset = dataset.batch(batch_size)



num_topics = 6              # K
num_words = len(vocab)      # V



doc_lengths = np.sum(counts, axis = 1)
doc_lengths = np.array(doc_lengths)
avg_doc_length = np.mean(doc_lengths)
doc_lengths_1d = doc_lengths
doc_lengths = tf.convert_to_tensor(doc_lengths)
doc_lengths_K = tf.concat([doc_lengths]*num_topics, axis = 1)
doc_lengths_K = tf.cast(doc_lengths_K, tf.float32)

########################################
### Define Data Generating Process #####
########################################

# document-topic distribution
# theta ~ Gamma(0.3, 0.3)

# neutral topic-word distribution
# beta_kv ~ Gamma(0.3, 0.3)

# adjusted topic-word distribution
# beta_tilde_kv ~ gamma(2, 1)



##################################
### Define Prior Hyperparameter ##
##################################


#num_documents, num_topics, num_words, num_authors = 10, 2, 44, 5
# parameter for theta - document-topic distribution
a_theta = tf.fill(dims = [num_documents, num_topics], value = model_hyperparameter["theta_shape"])
b_theta = tf.fill(dims = [num_documents, num_topics], value = model_hyperparameter["theta_rate"])
# parameter for beta - topic-word distribution
a_beta = tf.fill(dims = [num_topics, num_words], value = model_hyperparameter["beta_shape"])
b_beta = tf.fill(dims = [num_topics, num_words], value = model_hyperparameter["beta_rate"])



#####################################################
#### Define Keywords + Beta Tilde Hyperparameter ####
#####################################################

kw_indices_topics = list()
not_in_vocab_words = list()

for idx, topic in enumerate(keywords.keys()):
    # idx indicates the topic index -> used for beta-tilde adjustments
    for keyword in keywords[topic]:
        try:
            kw_index = list(vocab).index(keyword)
            kw_indices_topics.append([idx, kw_index])
        except Exception:
            print(f"###keyword: {keyword} from topic {topic} not in vocabulary - > will be pruned")
            # keywords[topic].remove(keyword) # GEHT NICHT WEIL DANN ÜBERSPRINGT ER IMMER DAS NÄCHSTE KEYWORD BEI DER NÄCHSTEN ITERATION!!!
            not_in_vocab_words.append(keyword)

# Remove keywords which are not in vocab!
for topic in keywords.keys():
    for na_word in not_in_vocab_words:
        try:
            keywords[topic].remove(na_word)
        except:
            pass

kw_indices = tf.convert_to_tensor(kw_indices_topics) # 72,2

# num_kw = len(list(itertools.chain(*keywords.values()))) # 75
# num_kw = len(kw_indices)
num_kw = len([kw for kws in keywords.values() for kw in kws])


## prepare
beta_tilde_a = tf.fill(dims = (num_kw), value = model_hyperparameter["beta_tilde_shape"]) # 75
beta_tilde_b = tf.fill(dims = (num_kw), value = model_hyperparameter["beta_tilde_rate"])
beta_tilde_a = tf.cast(beta_tilde_a, "float32")
beta_tilde_b = tf.cast(beta_tilde_b, "float32")


####################################
### Define Data Generating Model ###
####################################

def generative_model(a_theta, b_theta , a_beta, b_beta ,
                     a_beta_tilde , b_beta_tilde ,
                     kw_indices , document_indices, doc_lengths):
    """
    keyATM as GaP model!
    """

    #### MODELLSPEZIFIKATION NACH POISSON MODELL VON BLEI:
    ## https://www.cs.toronto.edu/~lcharlin/papers/GopalanCharlinBlei_nips14.pdf

    # Theta over Documents (and topics)
    theta = yield tfd.Gamma(a_theta, b_theta, name = "document_topic_distribution")

    # Beta over Topics
    beta = yield tfd.Gamma(a_beta, b_beta, name = "topic_word_distribution")
    beta_tilde = yield tfd.Gamma(a_beta_tilde, b_beta_tilde, name = "adjusted_topic_word_distribution")

    beta_star = tf.tensor_scatter_nd_add(beta, kw_indices, beta_tilde)


    theta_batch = tf.gather(theta, document_indices)

    # Scale theta by document length (according to GaP paper)
    relevant_doc_lengths = tf.gather(doc_lengths, document_indices)
    theta_bar = theta_batch / relevant_doc_lengths

    rate = tf.matmul(theta_bar, beta_star)

    y = yield tfd.Poisson(rate=rate, name="word_count")



def create_prior_batched(document_indices):
    model_joint = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda: generative_model(a_theta, b_theta, a_beta, b_beta, beta_tilde_a, beta_tilde_b, kw_indices,
                                 document_indices, doc_lengths_K)
    )
    return model_joint





def model_joint_log_prob(theta, beta, beta_tilde, document_indices, counts):
    model_joint = create_prior_batched(document_indices)
    return model_joint.log_prob_parts([theta, beta, beta_tilde, counts])



## surrogate posterior

# theta - document distribution
a_theta_S = tfp.util.TransformedVariable(
    tf.fill([num_documents, num_topics], model_hyperparameter["t_S_S"]),
    bijector=tfp.bijectors.Softplus(),
    name="theta_shape"
)
b_theta_S = tfp.util.TransformedVariable(
    tf.fill([num_documents, num_topics], model_hyperparameter["t_R_S"]),
    bijector=tfp.bijectors.Softplus(),
    name="theta_rate"
)


# beta - neutral topics / objective topic distribution
a_beta_S = tfp.util.TransformedVariable(
    tf.fill([num_topics, num_words], model_hyperparameter["b_S_S"]),
    bijector=tfp.bijectors.Softplus(),
    name="beta_shape"
)
b_beta_S = tfp.util.TransformedVariable(
    tf.fill([num_topics, num_words], model_hyperparameter["b_R_S"]),
    bijector=tfp.bijectors.Softplus(),
    name="beta_rate"
)


# beta_tilde - adjusted topic distribution (seed)
a_beta_tilde_S = tfp.util.TransformedVariable(
    tf.fill([num_kw], 2.0),
    bijector=tfp.bijectors.Softplus(),
    name="beta_tilde_shape"
)
b_beta_tilde_S = tfp.util.TransformedVariable(
    tf.fill([num_kw], 1.0),
    bijector=tfp.bijectors.Softplus(),
    name="beta_tilde_rate"
)

@tfd.JointDistributionCoroutineAutoBatched
def variational_family():
    theta_surrogate = yield tfd.Gamma(a_theta_S, b_theta_S, name = "theta_surrogate")
    beta_surrogate = yield tfd.Gamma(a_beta_S, b_beta_S, name="beta_surrogate")
    beta_tilde = yield tfd.Gamma(a_beta_tilde_S, b_beta_tilde_S, name = "beta_tilde_surrogate")

variational_family_surrogates = variational_family
len(variational_family_surrogates.trainable_variables)



def recode_theta(number):
    if number == 0:
        return "pet supplies"
    if number == 1:
        return "toys games"
    if number == 2:
        return "beauty"
    if number == 3:
        return "baby products"
    if number == 4:
        return "health personal care"
    if number == 5:
        return "grocery gourmet food"


#### PERFORM VI


# lr scheduler function
def power_scheduling(epoch, steps = 200 , initial_lr = 0.1):
    """
    LR is a function of the iteration number t: eta(t) = eta_0 / (1+t/s)^c.

    The initial learning rate eta_0, the power c (typical set to 1) and the steps s are
    hyperparameters.

    The higher the steps the more linear the decrease in the lr
    """

    # steps = epoch * steps

    return initial_lr / (1+epoch/steps)**1

# Initialize lr scheduler
# power_scheduling_fn = power_scheduling(steps = 10, initial_lr= 20)



def piecewise_constant_scheduling(epoch):
    """
    Use a constant learning rate for a number of epochs (e.g., η0 = 0.1 for 5 epochs),
    then a smaller learning rate for another number of epochs (e.g., η1 = 0.001 for
    50 epochs), and so on. Although this solution can work very well, it requires
    fiddling around to figure out the right sequence of learning rates and how long to
    use each of them.

    # 5k model: 0.1, 0.01, 0.001
    """

    if epoch < 250:
        return 0.001
    elif epoch < 650:
        return 0.0001
    else:
        return 0.00001



optim = tf.optimizers.Adam(learning_rate = model_hyperparameter["learning_rate"])
losses = list()
entropys = []
log_priors = list()
epoch_losses = list()
epoch_entropys = list()
epoch_log_priors = list()
variational_parameters = list() # store variational parameters
epoch_accuracy = list()
epoch_reconstruction_losses = list()

@tf.function
def train_step(inputs, outputs, optim):
    document_indices = inputs["document_indices"]

    # Use gradient tape for reverse-mode automatic differentiation
    with tf.GradientTape() as tape:

        # Sample from variational family - q(theta, beta, beta_tilde)
        theta, beta, beta_tilde = variational_family_surrogates.sample()

        # Compute log likelihood and entropy loss
        log_prior_losses = model_joint_log_prob(theta, beta, beta_tilde, document_indices, tf.sparse.to_dense(outputs))
        log_prior_loss_theta, log_prior_loss_beta, log_prior_loss_beta_tilde, reconstruction_loss = log_prior_losses
        # log_prior_old = tf.reduce_sum(log_prior_losses)

        # rescale reconstruction loss since it is only based on a mini-batch
        # recon_scaled = reconstruction_loss * (tf.cast(tf.shape(outputs)[0], tf.float32)/tf.cast(tf.constant(num_documents), tf.float32))
        recon_scaled = reconstruction_loss * tf.dtypes.cast(tf.constant(num_documents) / tf.shape(outputs)[0], tf.float32)
        log_prior = tf.reduce_sum([log_prior_loss_theta, log_prior_loss_beta, log_prior_loss_beta_tilde, recon_scaled])
        entropy = variational_family_surrogates.log_prob(theta, beta, beta_tilde)

        # Calculate negative elbo
        neg_elbo = - tf.reduce_mean(log_prior - entropy)

    # Reparametrize
    trainable_variables = tape.watched_variables()
    grads = tape.gradient(neg_elbo, trainable_variables)
    optim.apply_gradients(zip(grads, trainable_variables))
    return neg_elbo, entropy, log_prior, recon_scaled


def progress_bar(progress, total, epoch_runtime=0):
    """
    According to: https://www.youtube.com/watch?v=x1eaT88vJUA
    """
    percent = 100 * (progress/float(total))
    bar = "*" * int(percent) + "-" * (100-int(percent))
    print(f"\r|{bar}| {percent:.2f}% [{epoch_runtime:.4f}/s per epoch]", end = "\r")


lrs = list()
num_epochs = model_hyperparameter["num_epochs"]


# progress_bar(0, num_epochs)
a_time = time.time()
for idx, epoch in enumerate(range(num_epochs)):

    start_time = time.time()
    epoch_loss = list()
    epoch_entropy = list()
    epoch_log_prior = list()
    epoch_reconstruction_loss = list()

    if model_hyperparameter["lr_scheduler"] == "power":
        optim.lr.assign(power_scheduling(epoch, steps = 20))
    if model_hyperparameter["lr_scheduler"] == "piecewise":
        optim.lr.assign(piecewise_constant_scheduling(epoch))

    if model_hyperparameter["lr_scheduler"] == "dynamic":
        """
        If the average percentage change over the last 50 epoch is smaller than a certain threshold,
        we half the lr in order to improve model training, i.e. reverse-mode automatic differentiation.
        """
        if epoch % 150 == 0:

            # Get last neg_elbos
            last_losses = epoch_losses[-75:]
            # Compute pct change
            loss_pct_change = np.abs(np.diff(last_losses)/last_losses[:-1])
            mean_loss_pct_change = np.mean(loss_pct_change)

            # Half learning rate
            if mean_loss_pct_change < 0.001:

                print(f"MEAN LOSS PCT CHANGE: {mean_loss_pct_change}")
                actual_lr = optim.lr.numpy()
                optim.lr.assign(actual_lr/2)

    lrs.append(optim.lr.numpy())


    if early_stopping == True:

        if epoch % 15 == 0:
            last_losses = epoch_losses[-15:]

            loss_pct_change = np.abs(np.diff(last_losses) / last_losses[:-1])
            mean_loss_pct_change = np.mean(loss_pct_change)

            if mean_loss_pct_change < 0.0005:
                break




    for batch_index, batch in enumerate(iter(dataset)):
        batches_per_epoch = len(dataset) # in dem beispiel batches_per_epoch == 5, da 48/10 = 4.8 -> 5
        step = batches_per_epoch * epoch + batch_index

        # define input (indices) and corresponding outputs
        inputs, outputs = batch

        # Calculate losses
        neg_elbo, entropy, prior_loss, recon_loss = train_step(inputs, outputs, optim)

        # Store results
        losses.append(neg_elbo.numpy())
        entropys.append(entropy)
        log_priors.append(prior_loss)
        epoch_loss.append(neg_elbo.numpy())
        epoch_entropy.append(entropy)
        epoch_log_prior.append(prior_loss)
        epoch_reconstruction_loss.append(recon_loss)

    end_time = time.time()
    epoch_losses.append(np.mean(epoch_loss))
    epoch_entropys.append(np.mean(epoch_entropy))
    epoch_log_priors.append(np.mean(epoch_log_prior))
    epoch_reconstruction_losses.append(np.mean(epoch_reconstruction_loss))
    # variational_parameters.append(
    #     [variable.read_value() for variable in variational_family_surrogates.trainable_variables])

    try:
        E_theta = a_theta_S / b_theta_S
        categories = np.argmax(E_theta, axis=1)
        df1["keyatm_label"] = categories
        df1["keyatm_label"] = df1["keyatm_label"].apply(recode_theta)
        df1["Accuracy"] = df1.Cat1 == df1.keyatm_label

        # Overall accuracy
        accuracy = np.sum(df1.Accuracy) / df1.shape[0]
        # print(f"- OVERALL ACCURACY: {accuracy:.4f} -")
        epoch_accuracy.append(accuracy)
    except Exception as msg:
        print(f"Cant compute accuracy bcs of: {msg}")
        epoch_accuracy.append(np.nan)

    # Update progress bar
    # progress_bar(idx + 1, num_epochs, end_time - start_time)


    # Print model fit each x epochs
    if epoch % 2 == 0:
        print(f"EPOCH: {epoch}")
        print("EPOCH runtime: {duration:.4f}".format(duration=end_time - start_time))
        print(f"LEARNING RATE: {optim.lr.numpy()}")
        print("     -----     ")
        print(f"NEG ELBO: {np.mean(epoch_loss)}")
        print(f"Entropy: {np.mean(epoch_entropy)}")
        print(f"log_prior: {np.mean(epoch_log_prior)}")
        print(f"Recon loss: {np.mean(recon_loss)}")

        print("----------------------------")
        print("--- VALIDATION RESULTS: ----")
        print(f"- OVERALL ACCURACY: {accuracy:.4f} -")
        # try:
        #     E_theta = a_theta_S / b_theta_S
        #     categories = np.argmax(E_theta, axis=1)
        #     df1["keyatm_label"] = categories
        #     df1["keyatm_label"] = df1["keyatm_label"].apply(recode_theta)
        #     df1["Accuracy"] = df1.Cat1 == df1.keyatm_label
        #
        #     # Overall accuracy
        #     accuracy = np.sum(df1.Accuracy) / df1.shape[0]
        #     print(f"- OVERALL ACCURACY: {accuracy:.4f} -")
        #     epoch_accuracy.append(accuracy)
        # except Exception as msg:
        #     print(f"Cant compute accuracy bcs of: {msg}")
        #     epoch_accuracy.append(np.nan)
        print("----------------------------")
        print("##############################")

b_time = time.time()
print(f"TOTAL TIME: {b_time - a_time}")


# plt.plot(losses)
# plt.show()


## -- MODEL EVALUATION --

# E[theta] = theta_shape/theta_rate
E_theta = a_theta_S/b_theta_S

# E[beta]
E_beta = a_beta_S/b_beta_S

# E[beta_tilde]
E_beta_tilde = a_beta_tilde_S/b_beta_tilde_S

# E[beta_star]
beta_star = tf.tensor_scatter_nd_add(E_beta, kw_indices, E_beta_tilde)

# See betas
betas = pd.DataFrame(tf.transpose(beta_star), index = vocab, columns = list(keywords.keys()))

thetas_softmax = tf.nn.softmax(E_theta)

## PRINT HOTTEST WORDS PER TOPIC
top_words = np.argsort(-beta_star, axis = 1)
topic_strings = list()
for topic_idx in range(num_topics):
    neutral_start_string = "{}:".format(list(keywords.keys())[topic_idx])
    words_per_topic = 50
    neutral_row = [vocab[word] for word in top_words[topic_idx, :words_per_topic]]
    neutral_row_string = ", ".join(neutral_row)
    neutral_string = " ".join([neutral_start_string, neutral_row_string])

    topic_strings.append(" \n".join(
        [neutral_string]
    ))
topic_strings



# Intensity/counts matrix - ins verhältnis setzen
# np.shape(counts.toarray())
# vocab_sums = np.sum(counts.toarray(), axis = 0) # vocab_sums
# vocab_sums = pd.DataFrame(vocab_sums, index = vocab)

# vocab_sums_exploded = np.transpose(np.tile(vocab_sums, (num_topics, 1)))
# betas_weighted_with_word_counts = np.divide(betas, vocab_sums_exploded)




# Calculate accuracy
categories = np.argmax(E_theta, axis=1)

df1["keyatm_label"] = categories

def recode_theta(number):
    if number == 0:
        return "pet supplies"
    if number == 1:
        return "toys games"
    if number == 2:
        return "beauty"
    if number == 3:
        return "baby products"
    if number == 4:
        return "health personal care"
    if number == 5:
        return "grocery gourmet food"


df1["keyatm_label"] = df1["keyatm_label"].apply(recode_theta)
df1["Accuracy"] = df1.Cat1 == df1.keyatm_label

# Overall accuracy
print(f"OVERALL ACCURACY: {np.sum(df1.Accuracy) / df1.shape[0]}")

# Accuracy per topic
cats = np.unique(df1.Cat1)
for cat in cats:
    df = df1[(df1.Cat1 == cat)]
    print(f"Accuracy topic {cat}: {np.sum(df.Accuracy) / df.shape[0]}")

# ----- Save results -----

if save_files == True:
    # FIXED EPOCH
    if early_stopping == False:
        df1.to_csv(f"./keyATM/paper_code/final_model_fits/VSTM/{variant}/{variant}_VSTM_df.csv")
        pd.DataFrame(E_theta).to_csv(f"./keyATM/paper_code/final_model_fits/VSTM/{variant}/{variant}_VSTM_theta.csv")
        pd.DataFrame(beta_star).to_csv(f"./keyATM/paper_code/final_model_fits/VSTM/{variant}/{variant}_VSTM_beta.csv")
        pd.DataFrame(a_beta_tilde_S).to_csv(
            f"./keyATM/paper_code/final_model_fits/VSTM/{variant}/{variant}_VSTM_a_beta_tilde_S.csv")
        pd.DataFrame(b_beta_tilde_S).to_csv(
            f"./keyATM/paper_code/final_model_fits/VSTM/{variant}/{variant}_VSTM_b_beta_tilde_S.csv")
        pd.DataFrame({"neg_elbo":epoch_losses, "entropy":epoch_entropys, "log_prior":epoch_log_priors,
                      "recon":epoch_reconstruction_losses, "accuracy": epoch_accuracy}).to_csv(
            f"./keyATM/paper_code/final_model_fits/VSTM/{variant}/{variant}_VSTM_loss_metrics.csv")

    # EARLY STOPPING
    if early_stopping == True:
        df1.to_csv(f"./keyATM/paper_code/final_model_fits/VSTM/early_stopping/{variant}/{variant}_VSTM_df.csv")
        pd.DataFrame(E_theta).to_csv(f"./keyATM/paper_code/final_model_fits/VSTM/early_stopping/{variant}/{variant}_VSTM_theta.csv")
        pd.DataFrame(beta_star).to_csv(f"./keyATM/paper_code/final_model_fits/VSTM/early_stopping/{variant}/{variant}_VSTM_beta.csv")
        pd.DataFrame({"neg_elbo": epoch_losses, "entropy": epoch_entropys, "log_prior": epoch_log_priors,
                      "recon": epoch_reconstruction_losses, "accuracy": epoch_accuracy}).to_csv(
            f"./keyATM/paper_code/final_model_fits/VSTM/early_stopping/{variant}/{variant}_VSTM_loss_metrics.csv")



# ----- Classification report -----
from sklearn.metrics import classification_report
import pprint
pprint.pprint(classification_report(df1.Cat1, df1.keyatm_label))
#
#
# # -- PLOT MODEL FIT --
# show_since = -num_epochs*len(dataset)+100
# fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 11))
# axs[0].set_title("Log Likelihood loss with fixed data")
# axs[0].plot(log_priors[show_since:])
# axs[0].set_ylabel("Log Likelihood")
# axs[0].grid(True)
# axs[1].set_title("Entropy loss")
# axs[1].plot(entropys[show_since:])
# axs[1].set_ylabel("Entropy")
# axs[1].grid(True)
# axs[2].set_title("Negative ELBO loss")
# axs[2].plot(losses[show_since:])
# axs[2].set_ylabel("Neg ELBO")
# axs[2].grid(True)
# axs[2].set_xlabel("Iteration")
# fig.suptitle("Variational Inference model losses")
# plt.show()
#
# ## ON EPOCH LEVEL ##
show_since = 0
fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 11))
axs[0].set_title("Log joint loss")
axs[0].plot(epoch_log_priors[show_since:], label = "log joint loss")
axs[0].plot(epoch_reconstruction_losses[show_since:], label = "reconstruction loss")
axs[0].set_ylabel("Log joint")
axs[0].legend()
axs[0].grid(True)
axs[1].set_title("Entropy loss")
axs[1].plot(epoch_entropys[show_since:])
axs[1].set_ylabel("Entropy")
axs[1].grid(True)
axs[2].set_title("Negative ELBO loss")
axs[2].plot(epoch_losses[show_since:])
axs[2].set_ylabel("Neg ELBO")
axs[2].grid(True)
axs[2].set_xlabel("Iteration")
fig.suptitle("VSTM inference loss")
plt.show()

#
# fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 15))
# doc0 = 410
# axs[0].bar(list(range(len(E_theta[doc0]))), E_theta[doc0])
# axs[0].set_title(f"Document-Topic distribution for document {doc0}")
# doc1 = 41
# axs[1].bar(list(range(len(E_theta[doc1]))), E_theta[doc1])
# doc2 = 100
# axs[1].set_title(f"Document-Topic distribution for document {doc1}")
# axs[2].bar(list(range(len(E_theta[doc2]))), E_theta[doc2])
# axs[2].set_title(f"Document-Topic distribution for document {doc2}")
# plt.show()
#
# #
# plt.plot(epoch_losses[0-model_hyperparameter["num_epochs"] : ])
# plt.title("Negative ELBO per EPOCH (average over batches)")
# plt.grid(True)
# plt.show()
# #

# GET LEARNING RATE CHANGES
unique_lrs = np.unique(lrs)
unique_lrs = unique_lrs[::-1]
lr_change_at_epoch = np.array([lrs.index(i) for i in unique_lrs])
unique_lrs = unique_lrs[1:]
lr_change_at_epoch = lr_change_at_epoch[1:]
### Accuracy vs Negative ELBO Loss per epoch

fig, ax1 = plt.subplots(figsize = (12,7))
plt.title(f"Loss & accuracy per elbo (sample size = {df1.shape[0]})")
plt.suptitle(f"{list(model_hyperparameter.items())[:5]}\n{list(model_hyperparameter.items())[5:]}")
color = "tab:blue"
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Negative Elbo Loss")
ax1.plot(epoch_losses[show_since-model_hyperparameter["num_epochs"]:], color = color)
ax1.tick_params(axis = "y", labelcolor = color)
ax1.xaxis.grid()
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Model accuracy", color = color)
ax2.plot(epoch_accuracy[show_since-model_hyperparameter["num_epochs"]:], color = color)
ax2.tick_params(axis = "y", labelcolor = color)
# ax2.annotate("LR change",
#              xy = (100, ax2.get_ylim()[0]), xytext=(100, ax2.get_ylim()[0]-0.05),
#              arrowprops = dict(facecolor = "black", shrink = 0.05))
for idx, i in enumerate(unique_lrs):
    ax2.annotate(f"LR: {i:.4f}",
                 xy = (lr_change_at_epoch[idx]+show_since, ax2.get_ylim()[0]),
                 xytext=(lr_change_at_epoch[idx]+show_since, ax2.get_ylim()[0]-0.05),
                 arrowprops = dict(facecolor = "black", shrink = 0.05, alpha = 0.3))

plt.show()





#
#
# # Document-Topic Distribution
# document_topic_distribution = E_theta.numpy()
# document_topic_distribution = pd.DataFrame(document_topic_distribution, columns=[f"topic {i}" for i in list(range(num_topics))])
#
# document_samples = document_topic_distribution.sample(10)
# plt.figure(figsize=(12, 7))
# for index, row in document_samples.iterrows():
#     plt.plot(row, label = f"Document {index}")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.title(f"Expected document-topic distribution for {len(document_samples)} randomly selected documents.")
# plt.show()
#
#
# # Learning rate
# plt.figure(figsize=(12,7))
# plt.title(f"Learning rate over epochs")
# plt.plot(lrs)
# plt.grid(True, alpha = 0.3)
# plt.show()

#
#
# #### diagnostics
# doc_lengths_1d # 1xD
# doc_lengths_K # DxK
# doc_lengths_V # DxV
#
# E_theta = E_theta
# avg_thetas = np.mean(E_theta, axis = 1)
# sum_thetas = np.sum(E_theta, axis = 1)
#
# E_theta_bar = E_theta / doc_lengths_K
# avg_thetas_bar = np.mean(E_theta_bar, axis = 1)
# sum_thetas_bar = np.sum(E_theta_bar, axis = 1)
#
# E_rate = tf.matmul(E_theta, beta_star)
# E_rate_bar = E_rate / doc_lengths_V
#
# sum_E_rate = np.sum(E_rate, axis = 1)
# sum_E_rate_bar = np.sum(E_rate_bar, axis = 1)

#
#
#
#
#
# stat_analysis = False
# if stat_analysis == True:
#     # Dokumentenlänge vs. durchschnittliches Theta
#     plt.scatter(doc_lengths_1d, avg_thetas)
#     plt.xlabel("Document length")
#     plt.ylabel("Average theta UNSCALED")
#     plt.title("Document length vs average theta per document UNSCALED")
#     plt.show()
#
#     # Dokumentenlänge vs. durchschnittliches Theta SKALIERT
#     plt.scatter(doc_lengths_1d, avg_thetas_bar)
#     plt.xlabel("Document length")
#     plt.ylabel("Average theta SCALED")
#     plt.title("Document length vs average theta per document SCALED")
#     plt.show()
#
#
#
#     # Dokumentenlänge vs. Summe Rate UNSCALED
#     plt.scatter(doc_lengths_1d, sum_E_rate)
#     plt.xlabel("Document length")
#     plt.ylabel("UNSCALED poisson rate")
#     plt.title("Document length vs UNSCALED poisson rate")
#     plt.show()
#
#     # Dokumentenlänge vs. durchschnittliches Theta SKALIERT
#     plt.scatter(doc_lengths_1d, sum_E_rate_bar)
#     plt.xlabel("Document length")
#     plt.ylabel("SCALED poisson rate")
#     plt.title("Document length vs SCALED poisson rate")
#     plt.show()
#

# abc = False
# if abc == True:
#     import seaborn as sns
#     # Heatmap thetas
#     thetas_sample = document_topic_distribution.sample(500)
#     thetas_sample_softmax = tf.nn.softmax(thetas_sample)
#     sns.heatmap(thetas_sample_softmax, cmap="binary").set(title = r"Document-topic distribution $q_{\theta}$")
#     # sns.heatmap(thetas_sample, vmin = 0.0, vmax = 1.0, cmap = "binary")
#     # sns.heatmap(thetas_sample, vmin = 0.0, vmax = 5.0, cmap = "binary")
#     plt.show()
#
#
#
#     ### MODEL METRICS: https://stackoverflow.com/questions/62327099/confusion-matrix-python
#     from sklearn.metrics import classification_report, confusion_matrix
#     import pprint
#     pprint.pprint(classification_report(df1.Cat1, df1.keyatm_label))
#
#     keys_plot = ["baby", "beauty", "grocery", "health", "pets", "toys"]
#     keys_plot.insert(0, " ")
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     conf_matrix = confusion_matrix(y_true = df1.Cat1, y_pred= df1.keyatm_label)
#     cax = ax.matshow(conf_matrix, cmap = "binary")
#     ax.set_xticklabels(keys_plot)
#     ax.set_yticklabels(keys_plot)
#     plt.show()
#
#
#
    # keys_plot = ["baby", "beauty", "grocery", "health", "pets", "toys"]
    # keys_plot.insert(0, " ")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # conf_matrix = confusion_matrix(y_true = df1.Cat1, y_pred= df1.keyatm_label)
    # cax = ax.matshow(conf_matrix, cmap = "binary")
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='medium')
    # ax.set_xticklabels(keys_plot)
    # ax.set_yticklabels(keys_plot)
    # plt.show()
    #
    # # DOUBLE CHECK THE LABELS FROM THE CONFUSION MATRIX
    # df1[(df1.Cat1 == "baby products") & (df1.keyatm_label == "pet toys")]

#
#
# # HEATMAP BETAS
# # vorlage: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# sub_df = betas.sort_values(by=["pets"], ascending=False).iloc[:30, :]
# fig, ax = plt.subplots()
# # plot raw data
# im = ax.imshow(sub_df)
#
# # show tick labels
# ax.set_xticks(np.arange(len(sub_df.columns)), labels = sub_df.columns)
# ax.set_yticks(np.arange(sub_df.shape[0]), labels = sub_df.index)
#
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
#
# plt.show()

# #
# #
# #
#
# # HEATMAP BETAS
# # vorlage: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# sub_df = betas.sort_values(by=["pets"], ascending=False).iloc[:30, :]
# sub_df = sub_df.T
# fig, ax = plt.subplots()
# # plot raw data
# im = ax.imshow(sub_df)
#
# # show tick labels
# ax.set_xticks(np.arange(sub_df.shape[1]), labels = sub_df.columns)
# ax.set_yticks(np.arange(len(sub_df.index)), labels = sub_df.index)
#
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
#
# plt.show()

# #
# plt.imshow(a_beta_S, b_beta_S)
# plt.show()








####################################
###### BOOTSTRAP EXPERIMENT ########
####################################

# # Wir samplen jetzt aus den gefitteten Werten einen eigenen corpus
# p_rate = tf.matmul(E_theta, beta_star)
# synthetic_y = tfd.Poisson(rate = p_rate)
# synthetic_y.sample()
#
# s_y = tfd.Poisson(rate = tf.matmul(variational_family_surrogates.sample_distributions()[0][0], variational_family_surrogates.sample_distributions()[0][1]))
# a1 = tfd.Normal([3,4,5], [1,1,1])
# a1.sample(2)
#
# np.save("./keyATM/paper_code/fitted_variational_parameter/e_theta.npy", E_theta)
# np.save("./keyATM/paper_code/fitted_variational_parameter/e_beta.npy", E_beta)
# np.save("./keyATM/paper_code/fitted_variational_parameter/e_beta_tilde.npy", E_beta_tilde)
# np.save("./keyATM/paper_code/fitted_variational_parameter/e_beta_star.npy", beta_star)
# np.save("./keyATM/paper_code/fitted_variational_parameter/e_rate.npy", p_rate)


# Save variational parameter for later import when generating synthetic data
# np.save("./keyATM/paper_code/fitted_variational_parameter/a_theta_S.npy", a_theta_S)
# np.save("./keyATM/paper_code/fitted_variational_parameter/b_theta_S.npy", b_theta_S)
# np.save("./keyATM/paper_code/fitted_variational_parameter/a_beta_S.npy", a_beta_S)
# np.save("./keyATM/paper_code/fitted_variational_parameter/b_beta_S.npy", b_beta_S)
# np.save("./keyATM/paper_code/fitted_variational_parameter/a_beta_tilde_S.npy", a_beta_tilde_S)
# np.save("./keyATM/paper_code/fitted_variational_parameter/b_beta_tilde_S.npy", b_beta_tilde_S)
# np.save("./keyATM/paper_code/fitted_variational_parameter/kw_indices.npy", kw_indices)

#
# # Load variables
# import tensorflow as tf
# import tensorflow_probability as tfp
# import numpy as np
# import pandas as pd
# import scipy.sparse as sparse
#
# tfd = tfp.distributions
#
# a_theta_S = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/a_theta_S.npy"))
# b_theta_S = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/b_theta_S.npy"))
# a_beta_S = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/a_beta_S.npy"))
# b_beta_S = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/b_beta_S.npy"))
# a_beta_tilde_S = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/a_beta_tilde_S.npy"))
# b_beta_tilde_S = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/b_beta_tilde_S.npy"))
# kw_indices = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/kw_indices.npy"))
#
# #
# # E_theta = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/e_theta.npy"))
# # E_beta = tf.Variable(np.load("./keyATM/paper_code/fitted_variational_parameter/e_beta.npy"))
# # #
# #
# # #E_rate = np.load("./keyATM/paper_code/fitted_variational_parameter/e_rate.npy")
# #
# # synthetic_y = tfd.Poisson(rate = tf.matmul(E_theta, E_beta))
# # synthetic_y.sample()
#
#
#
# @tfd.JointDistributionCoroutineAutoBatched
# def synthetic_model():
#     """
#     keyATM as GaP model!
#     """
#
#     #### MODELLSPEZIFIKATION NACH POISSON MODELL VON BLEI:
#     ## https://www.cs.toronto.edu/~lcharlin/papers/GopalanCharlinBlei_nips14.pdf
#
#     # Theta over Documents (and topics)
#     theta = yield tfd.Gamma(a_theta_S, b_theta_S, name = "document_topic_distribution")
#
#     # Beta over Topics
#     beta = yield tfd.Gamma(a_beta_S, b_beta_S, name = "topic_word_distribution")
#     beta_tilde = yield tfd.Gamma(a_beta_tilde_S, b_beta_tilde_S, name = "adjusted_topic_word_distribution")
#
#     # linear combination of beta and beta_tilde. Update beta with beta_tilde values at kw_indices
#     beta_star = tf.tensor_scatter_nd_add(beta, kw_indices, beta_tilde)
#
#     # compute poisson rate
#     rate = tf.matmul(theta, beta_star)
#
#     # Calculate word count (DTM)
#     y = yield tfd.Poisson(rate=rate, name="word_count")
#
#
# synthetic_y = synthetic_model
#
#
# synthetic_y.sample()
#
# synthetic_y.sample_distributions()[0][0]
#
#
#
#
# @tfd.JointDistributionCoroutineAutoBatched
# def synthetic_model():
#     """
#     keyATM as GaP model!
#     """
#
#     # compute poisson rate
#     rate = tf.matmul(E_theta, E_beta)
#
#     # Calculate word count (DTM)
#     y = yield tfd.Poisson(rate=rate, name="word_count")
#
#
# synthetic_y = synthetic_model
#
#
# abc = sparse.csr_matrix(synthetic_y.sample().numpy(), dtype = np.float32)
#
#
# gpu_devices = tf.config.list_physical_devices('GPU')
# if gpu_devices:
#   tf.config.experimental.get_memory_usage('GPU:0')
#
#
# tf.config.experimental.get_memory_info('GPU:0')
#

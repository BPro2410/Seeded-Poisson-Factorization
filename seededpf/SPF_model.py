# Imports

import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.stats as stats
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.feature_extraction.text import CountVectorizer
from seededpf.SPF_helper import SPF_helper

# Shortcuts
# np.set_printoptions(suppress=True)
# warnings.filterwarnings("ignore")
tfd = tfp.distributions
tfb = tfp.bijectors

# seededpf class
class SPF(tf.keras.Model):
    """
    Tensorflow implementation of the Seeded Poisson Factorization topic model.
    """

    def __init__(self, keywords: dict,
                 residual_topics: int = 0):
        """
        Initialization of the seededpf object.

        :param keywords: Dictionary containing topics (keys) and keywords (values) of the form
        {'topic_1':['word1', 'word2'],'topic_2':['word1', 'word2']}
        :param residual_topics: Number of residual topics (i.e. topics with no prior information available)
        """

        super(SPF, self).__init__()

        # Initialize model parameters and check if keywords are defined properly
        self.__keywords = SPF_helper._check_keywords(keywords, residual_topics)
        self.residual_topics = residual_topics
        self.model_settings = {"num_topics": len(self.__keywords.keys()) + residual_topics}

        if not tf.config.list_physical_devices("GPU"):
            warnings.warn("No GPU support for your tensorflow version! Consider manual installing tensorflow with gpu "
                          f"support if possible. Tensorflow version: {tf.__version__}.")

    @property
    def keywords(self):
        return self.__keywords

    @keywords.setter
    def keywords(self, new_keywords):
        raise Exception("Please reinitialize the model with new keywords. No overwriting allowed!")

    def read_docs(self,
                  text: list[str],
                  count_vectorizer=CountVectorizer(stop_words="english", min_df=2),
                  batch_size: int = 1024,
                  seed: int = 2410):
        """
        Reads documents, processes them into the format required by the seededpf model and creates additional metadata.

        :param text: Text to be classified. Format: Either a list of strings or pd.Series.
        :param count_vectorizer: CountVectorizer object used to create the DTM.
        :param batch_size: Batch_size used for training.
        :param seed: Seed used for shuffeling the data.
        :return: None
        """

        # Initialize vectorizer
        cv = count_vectorizer
        cv.fit(text)

        # Create DTM
        counts = sparse.csr_matrix(cv.transform(text), dtype=np.float32)

        # Check if there are documents with 0 tokens
        zero_idx = np.where(np.sum(counts, axis=1) == 0)
        if len(zero_idx[0]) > 0:
            raise ValueError(f"There are documents with zero words after tokenization. "
                             f"Please remove them or adjust the tokenizer. Documents: {zero_idx[0]}.")

        # Add metadata
        self.model_settings["vocab"] = cv.get_feature_names_out()
        self.model_settings["num_words"] = len(self.model_settings["vocab"])
        self.model_settings["num_documents"] = counts.shape[0]
        self.model_settings["doc_length_K"] = tf.concat([counts.sum(axis = 1)] *
                                                        self.model_settings["num_topics"], axis=1)

        # Check if seed words are contained in vocabulary & create keyword indices tensor for beta-tilde
        # Remove keywords that do not occur from the keywords dictionary
        self.kw_indices_topics = list()
        not_in_vocab_words = list()

        for idx, topic in enumerate(self.__keywords.keys()):
            # idx indicates the topic index. Used for beta-tilde adjustments.
            for keyword in self.__keywords[topic]:
                try:
                    kw_index = list(self.model_settings["vocab"]).index(keyword)
                    self.kw_indices_topics.append([idx, kw_index])
                except Exception:
                    print(f"NOTE: The seed word '{keyword}' defined for topic '{topic}' is not in the vocabulary. "
                          f"Keyword dictionary will be pruned.")
                    not_in_vocab_words.append(keyword)

        # Remove keywords which are not in vocab!
        for topic in self.__keywords.keys():
            for na_word in not_in_vocab_words:
                try:
                    self.__keywords[topic].remove(na_word)
                except:
                    pass

        self.kw_indices_topics = tf.convert_to_tensor(self.kw_indices_topics)

        # Create tensorflow dataset
        random_state = np.random.RandomState(seed)
        documents = random_state.permutation(self.model_settings["num_documents"])
        shuffled_counts = counts[documents]
        count_values = shuffled_counts.data

        shuffled_counts = tf.SparseTensor(
            indices=np.array(shuffled_counts.nonzero()).T,
            values=count_values,
            dense_shape=shuffled_counts.shape
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            ({"document_indices": documents},
             shuffled_counts)
        )

        self.dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        print(f"DTM created with: {counts.shape[0]} documents and {counts.shape[1]} unique words!")


    def __create_model_parameter(self):
        """
        Creates both the prior parameter and variational parameter.
        :return: None
        """

        # --- Create prior parameter ---
        self.prior_parameter = dict()

        # Theta prior parameter - document-topic distribution
        self.prior_parameter["a_theta"] = tf.fill(
            dims=[self.model_settings["num_documents"], self.model_settings["num_topics"]],
            value=self.prior_params["theta_shape"])

        self.prior_parameter["b_theta"] = tf.fill(
            dims=[self.model_settings["num_documents"], self.model_settings["num_topics"]],
            value=self.prior_params["theta_rate"])

        # Beta prior parameter - topic-word distribution
        self.prior_parameter["a_beta"] = tf.fill(
            dims=[self.model_settings["num_topics"], self.model_settings["num_words"]],
            value=self.prior_params["beta_shape"])

        self.prior_parameter["b_beta"] = tf.fill(
            dims=[self.model_settings["num_topics"], self.model_settings["num_words"]],
            value=self.prior_params["beta_rate"])

        # Beta_tilde prior parameter - seed words
        self.num_kw = len([kw for kws in self.__keywords.values() for kw in kws])
        self.prior_parameter["a_beta_tilde"] = tf.fill(dims=[self.num_kw], value=self.prior_params["beta_tilde_shape"])
        self.prior_parameter["b_beta_tilde"] = tf.fill(dims=[self.num_kw], value=self.prior_params["beta_tilde_rate"])

        # --- Create free variational family parameter ---
        self.variational_parameter = dict()

        # theta - document distribution
        self.variational_parameter["a_theta_S"] = tfp.util.TransformedVariable(
            tf.fill([self.model_settings["num_documents"], self.model_settings["num_topics"]],
                    self.variational_params["theta_shape_S"]),
            bijector=tfp.bijectors.Softplus(),
            name="theta_shape"
        )
        self.variational_parameter["b_theta_S"] = tfp.util.TransformedVariable(
            tf.fill([self.model_settings["num_documents"], self.model_settings["num_topics"]],
                    self.variational_params["theta_rate_S"]),
            bijector=tfp.bijectors.Softplus(),
            name="theta_rate"
        )

        # beta - neutral topics / objective topic distribution
        self.variational_parameter["a_beta_S"] = tfp.util.TransformedVariable(
            tf.fill([self.model_settings["num_topics"], self.model_settings["num_words"]],
                    self.variational_params["beta_shape_S"]),
            bijector=tfp.bijectors.Softplus(),
            name="beta_shape"
        )
        self.variational_parameter["b_beta_S"] = tfp.util.TransformedVariable(
            tf.fill([self.model_settings["num_topics"], self.model_settings["num_words"]],
                    self.variational_params["beta_rate_S"]),
            bijector=tfp.bijectors.Softplus(),
            name="beta_rate"
        )

        # beta_tilde - adjusted topic distribution (seed)
        self.variational_parameter["a_beta_tilde_S"] = tfp.util.TransformedVariable(
            tf.fill([self.num_kw], self.variational_params["beta_tilde_shape_S"]),
            bijector=tfp.bijectors.Softplus(),
            name="beta_tilde_shape"
        )
        self.variational_parameter["b_beta_tilde_S"] = tfp.util.TransformedVariable(
            tf.fill([self.num_kw], self.variational_params["beta_tilde_rate_S"]),
            bijector=tfp.bijectors.Softplus(),
            name="beta_tilde_rate"
        )

    def __create_variational_family(self):
        """
        Creates the variational family.
        :return: Variational family object.
        """

        @tfd.JointDistributionCoroutineAutoBatched
        def variational_family(keywords=self.keywords):
            theta_surrogate = yield tfd.Gamma(
                self.variational_parameter["a_theta_S"],
                self.model_settings["doc_length_K"] * self.variational_parameter["b_theta_S"],
                name="theta_surrogate")
            beta_surrogate = yield tfd.Gamma(
                self.variational_parameter["a_beta_S"], self.variational_parameter["b_beta_S"],
                name="beta_surrogate")
            if len(keywords) != 0:
                beta_tilde = yield tfd.Gamma(
                    self.variational_parameter["a_beta_tilde_S"], self.variational_parameter["b_beta_tilde_S"],
                    name="beta_tilde_surrogate")

        return variational_family

    def __create_prior_batched(self, document_indices):
        """
        Definition of the data generating model.

        :param document_indices: Document indices. Relevant for batches.
        :return: Generative model object.
        """

        def generative_model(a_theta, b_theta, a_beta, b_beta,
                             a_beta_tilde, b_beta_tilde,
                             kw_indices, document_indices, doc_lengths, keywords):
            """
            Generative model of the seededpf!
            """

            # Theta over documents
            theta = yield tfd.Gamma(a_theta, b_theta, name="document_topic_distribution")

            # Beta over Topics
            beta = yield tfd.Gamma(a_beta, b_beta, name="topic_word_distribution")

            # Get batch of thetas
            theta_batch = tf.gather(theta, document_indices)

            # Compute Poisson rate
            if len(keywords) != 0:
                beta_tilde = yield tfd.Gamma(a_beta_tilde, b_beta_tilde, name="adjusted_topic_word_distribution")
                y = yield tfd.Poisson(
                    rate=tf.matmul(
                        theta_batch,
                        tf.tensor_scatter_nd_add(beta, kw_indices, beta_tilde)
                    ),
                    name="word_count"
                )
            else:
                y = yield tfd.Poisson(
                    rate=tf.matmul(
                        theta_batch,
                        beta
                    ),
                    name="word_count"
                )

        model_joint = tfp.distributions.JointDistributionCoroutineAutoBatched(
            lambda: generative_model(self.prior_parameter["a_theta"], self.prior_parameter["b_theta"],
                                     self.prior_parameter["a_beta"], self.prior_parameter["b_beta"],
                                     self.prior_parameter["a_beta_tilde"], self.prior_parameter["b_beta_tilde"],
                                     self.kw_indices_topics, document_indices, self.model_settings["doc_length_K"],
                                     self.keywords)
        )

        return model_joint

    def __model_joint_log_prob(self, theta, beta, beta_tilde, document_indices, counts):
        """
        Prior loss function.
        :param theta: q(\theta) sample.
        :param beta: q(\beta) sample.
        :param beta_tilde: q(\beta_tilde) sample.
        :param document_indices: Document indices relevant for batches
        :param counts: DTM counts (batched).
        :return: Log prior loss.
        """
        if len(self.keywords) != 0:
            model_joint = self.__create_prior_batched(document_indices)
            return model_joint.log_prob_parts([theta, beta, beta_tilde, counts])
        else:
            model_joint = self.__create_prior_batched(document_indices)
            return model_joint.log_prob_parts([theta, beta, counts])

    def model_train(self, lr: float = 0.1,
                    epochs: int = 500,
                    tensorboard: bool = False,
                    log_dir: str = os.getcwd(),
                    save_every: int = 1,
                    early_stopping=False,
                    pi=25,
                    delta=0.0005,
                    priors: dict = {},
                    variational_parameter: dict = {},
                    print_information=True,
                    print_progressbar=False):
        """
        Model training.

        :param lr: Learning rate for Adam optimizer.
        :param epochs: Model iterations.
        :param tensorboard: Indicator whether tensorflow logs should be saved.
        :param log_dir: Directory for the tensorboard logs.
        :param save_every: Tensorboard log interval.
        :param early_stopping: Bool, indicating whether early stopping should be activated or not.
        :param pi: Interval of epochs that should be watched for early stopping mechanism.
        :param delta: Convergence threshold for the early stopping mechanism.
        :param priors: Dictionary containing the prior parameter.
        :param variational_parameter: Dictionary containing the initial variational_parameter.
        :param print_information: Bool wheter information about training loss should be printed.
        :param print_progressbar: Bool, indicating whether a progressbar for training steps should be printed.
        :return: None
        """

        # Check prior parameter and variational parameter values
        self.prior_params = SPF_helper._check_priors(priors)
        self.variational_params = SPF_helper._check_variational_parameter(
            variational_parameter,
            corpus_info=self.model_settings["num_documents"])

        # Create all the required model parameters in matching dimensions
        self.__create_model_parameter()

        # Define the variational family
        self.variational_family = self.__create_variational_family()

        # Set optimizer
        optim = tf.optimizers.Adam(learning_rate=lr)

        def progress_bar(progress, total, epoch_runtime=0, neg_elbo = 0):
            """
            Simple progress bar to visualize the model runtime.
            """
            percent = 100 * (progress / float(total))
            bar = "*" * int(percent) + "-" * (100 - int(percent))
            print(f"\r|{bar}| {percent:.2f}% [{epoch_runtime:.4f}/s per epoch | Negative ELBO: {neg_elbo}]", end="\r")

        @tf.function
        @tf.autograph.experimental.do_not_convert
        def train_step(inputs, outputs, optim):
            """
            Train step using Tensorflows gradient tape.
            :param inputs: Document indices (via TF's batched dataset object)
            :param outputs: DTM counts (batched)
            :param optim: Optimizer
            :return: Model losses
            """

            document_indices = inputs["document_indices"]

            with tf.GradientTape() as tape:

                if len(self.keywords) != 0:
                    # Sample from the variational family
                    theta, beta, beta_tilde = self.variational_family.sample()

                    # Compute log prior loss
                    log_prior_losses = self.__model_joint_log_prob(
                        theta, beta, beta_tilde, document_indices, tf.sparse.to_dense(outputs)
                    )
                    log_prior_loss_theta, log_prior_loss_beta, \
                        log_prior_loss_beta_tilde, reconstruction_loss = log_prior_losses
                else:
                    # Compute the same except beta_tilde
                    theta, beta = self.variational_family.sample()

                    log_prior_losses = self.__model_joint_log_prob(
                        theta=theta, beta=beta, beta_tilde=None, document_indices=document_indices,
                        counts=tf.sparse.to_dense(outputs)
                    )

                    log_prior_loss_theta, log_prior_loss_beta, reconstruction_loss = log_prior_losses

                # Rescale reconstruction loss since it is only based on a mini-batch
                recon_scaled = reconstruction_loss * tf.dtypes.cast(
                    tf.constant(self.model_settings["num_documents"]) / (tf.shape(outputs)[0]),
                    tf.float32)

                if len(self.keywords) != 0:
                    log_prior = tf.reduce_sum(
                        [log_prior_loss_theta, log_prior_loss_beta, log_prior_loss_beta_tilde, recon_scaled])

                    # Compute entropy loss
                    entropy = self.variational_family.log_prob(theta, beta, beta_tilde)
                else:
                    log_prior = tf.reduce_sum(
                        [log_prior_loss_theta, log_prior_loss_beta, recon_scaled]
                    )
                    entropy = self.variational_family.log_prob(theta, beta)

                # Calculate negative elbo
                neg_elbo = - tf.reduce_mean(log_prior - entropy)

            # Reparametrize
            grads = tape.gradient(neg_elbo, tape.watched_variables())
            optim.apply_gradients(zip(grads, tape.watched_variables()))
            return neg_elbo, entropy, log_prior, recon_scaled

        # Log to tensorboard
        if tensorboard == True:
            summary_writer = tf.summary.create_file_writer(log_dir)
            summary_writer.set_as_default()

        # Start model training
        if print_progressbar == True:
            progress_bar(0, epochs)

        # Model metrics to save results
        self.model_metrics = dict(
            neg_elbo_loss=list(),
            recon_loss=list(),
            prior_loss=list(),
            entropy_loss=list()
        )

        # Start iterating
        for idx, epoch in enumerate(range(epochs)):
            # Capture epoch metrics
            start_time = time.time()
            epoch_loss = list()
            epoch_entropy = list()
            epoch_log_prior = list()
            epoch_reconstruction_loss = list()

            # Iterate through batches
            for batch_index, batch in enumerate(iter(self.dataset)):
                batches_per_epoch = len(self.dataset)
                step = batches_per_epoch * epoch + batch_index

                inputs, outputs = batch  # inputs = {'document_indices':[]}, outputs = counts

                # Calculate loss & reparametrize
                neg_elbo, entropy, prior_loss, recon_loss = train_step(inputs, outputs, optim)

                # Store batch loss in epoch loss
                epoch_loss.append(neg_elbo)
                epoch_entropy.append(entropy)
                epoch_log_prior.append(prior_loss)
                epoch_reconstruction_loss.append(recon_loss)

            # Store epoch results
            self.model_metrics["neg_elbo_loss"].append(np.mean(epoch_loss))
            self.model_metrics["recon_loss"].append(np.mean(epoch_reconstruction_loss))
            self.model_metrics["entropy_loss"].append(np.mean(epoch_entropy))
            self.model_metrics["prior_loss"].append(np.mean(epoch_log_prior))
            end_time = time.time()

            if print_information == True:
                # -- Print model stats --
                print("EPOCH: {} -- Total loss: {:.1f} -- Reconstruction loss: {:.1f} -- "
                      "Prior loss: {:.1f} -- Entropy loss: {:.1f}".format(epoch,
                                                                          self.model_metrics["neg_elbo_loss"][-1],
                                                                          self.model_metrics["recon_loss"][-1],
                                                                          self.model_metrics["prior_loss"][-1],
                                                                          self.model_metrics["entropy_loss"][-1]))

            if tensorboard == True and epoch % save_every == 0:
                tf.summary.text("topics", self.print_topics(), step=epoch)
                tf.summary.scalar("elbo/Negative ELBO", self.model_metrics["neg_elbo_loss"][-1], step=epoch)
                tf.summary.scalar("elbo/Entropy loss", self.model_metrics["entropy_loss"][-1], step=epoch)
                tf.summary.scalar("elbo/Log prior loss", self.model_metrics["prior_loss"][-1], step=epoch)
                tf.summary.scalar("elbo/Reconstruction loss", self.model_metrics["recon_loss"][-1], step=epoch)
                tf.summary.histogram("params/Theta shape surrogate",
                                     self.variational_parameter["a_theta_S"], step=epoch)
                tf.summary.histogram("params/Theta rate surrogate",
                                     self.variational_parameter["b_theta_S"], step=epoch)
                tf.summary.histogram("params/Beta shape surrogate",
                                     self.variational_parameter["a_beta_S"], step=epoch)
                tf.summary.histogram("params/Beta rate surrogate",
                                     self.variational_parameter["b_beta_S"], step=epoch)
                tf.summary.histogram("params/Beta tilde shape surrogate",
                                     self.variational_parameter["b_beta_tilde_S"], step=epoch)
                tf.summary.histogram("params/Beta tilde rate surrogate",
                                     self.variational_parameter["b_beta_tilde_S"], step=epoch)
                summary_writer.flush()

            if print_progressbar == True:
                progress_bar(idx + 1, epochs, end_time - start_time, self.model_metrics["neg_elbo_loss"][-1])

            # Check if early stopping criterion is met
            if early_stopping == True:
                if epoch % pi == 0:
                    last_losses = self.model_metrics["neg_elbo_loss"][-pi:]
                    loss_pct_change = np.abs(np.diff(last_losses) / last_losses[:-1])
                    mean_loss_pct_change = np.mean(loss_pct_change)
                    if mean_loss_pct_change < delta:
                        break

    def return_topics(self):
        """
        Calculate topic mean intensities and recode to the topics.
        :return: (Estimated topics, theta vector)
        """
        # Compute posterior means
        E_theta = self.variational_parameter["a_theta_S"] / self.variational_parameter["b_theta_S"]
        categories = np.argmax(E_theta, axis=1)

        def recode_cats(i):
            if i in list(range(len(self.keywords.keys()))):
                return list(self.keywords.keys())[i]
            else:
                return "No_keyword_topic_" + str(i - np.max(range(len(self.keywords))))

        topics = [recode_cats(i) for i in categories]
        return topics, E_theta

    def plot_model_loss(self):
        """
        Plots the model loss to check convergence.
        :return: None
        """

        fig, ax1 = plt.subplots(figsize=(12, 7))
        plt.title(f"seededpf loss plot on {self.model_settings['num_documents']} documents",
                  fontsize=15, weight="bold", color="0.2")
        ax1.set_xlabel("Epoch", fontsize=13, color="0.2")
        ax1.set_ylabel("Negative ELBO loss", fontsize=15, color="0.2")
        lns1 = ax1.plot(self.model_metrics["neg_elbo_loss"], color="black", label="Negative ELBO", lw=2.5, mec="w",
                        mew="2", alpha=0.9)
        lines = lns1
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=1, frameon=False, labelcolor="0.2",
                   prop={"weight": "bold", "size": 13})
        for axis in ["bottom", "left"]:
            ax1.spines[axis].set_linewidth(2.5)
            ax1.spines[axis].set_color("0.2")
        ax1.tick_params(width=2.5, labelsize=13)
        fig.tight_layout()
        plt.show()


    def calculate_topic_word_distributions(self):
        """
        Calculate posterior means for the topic-word distribution.
        :return: Topic-word distribution dataframe.
        """
        E_beta = self.variational_parameter["a_beta_S"] / self.variational_parameter["b_beta_S"]
        if len(self.keywords) != 0:
            E_beta_tilde = self.variational_parameter["a_beta_tilde_S"] / self.variational_parameter["b_beta_tilde_S"]
            beta_star = tf.tensor_scatter_nd_add(E_beta, self.kw_indices_topics, E_beta_tilde)
        else:
            beta_star = E_beta
        topic_names = list(self.keywords.keys())
        rs_names = [f"residual_topic_{i + 1}" for i in range(self.residual_topics)]
        return pd.DataFrame(tf.transpose(beta_star), index=self.model_settings["vocab"], columns=topic_names + rs_names)

    def print_topics(self, num_words: int = 50):
        """
        Prints the words with the highest mean intensity per topic.
        :param num_words: Number of words printed per topic.
        :return: Dictionary containing the most important words for each topic.
        """
        E_beta = self.variational_parameter["a_beta_S"] / self.variational_parameter["b_beta_S"]
        if len(self.keywords) != 0:
            E_beta_tilde = self.variational_parameter["a_beta_tilde_S"] / self.variational_parameter["b_beta_tilde_S"]
            beta_star = tf.tensor_scatter_nd_add(E_beta, self.kw_indices_topics, E_beta_tilde)
        else:
            beta_star = E_beta
        top_words = np.argsort(-beta_star, axis=1)

        hot_words = dict()

        for topic_idx in range(self.model_settings["num_topics"]):
            if topic_idx in list(range(len(self.keywords.keys()))):
                topic_name = "{}".format(list(self.keywords.keys())[topic_idx])
                words_per_topic = num_words
                hot_words_topic = [self.model_settings["vocab"][word] for word in
                                   top_words[topic_idx, :words_per_topic]]
                hot_words[topic_name] = hot_words_topic
            else:
                words_per_topic = num_words
                hot_words[f"Residual_topic_{topic_idx - len(self.keywords) + 1}"] = \
                    [self.model_settings["vocab"][word] for word in top_words[topic_idx, :words_per_topic]]

        return hot_words

    def plot_seeded_topic_distribution(self, topic: str,
                                       x_max: int = 10,
                                       detail: bool =False):
        """
        Plots the variational topic word distribution of all seed words belonging to the topic parameter.

        :param topic: Topic name whose variational seeded topic-word distribution should be plotted.
        :param x_max: Maximal value of the x axis
        :param detail: Whether the parameter of the variational gamma distributions should be printed in the legend.
        """
        topic_names = list(self.keywords.keys())
        rs_names = [f"residual_topic_{i + 1}" for i in range(self.residual_topics)]
        if topic not in topic_names + rs_names:
            raise ValueError(f"{topic} is not a valid topic name. Topic names are: {topic_names + rs_names}")

        kws_per_topic = [(idx, topic, len(kws)) for idx, (topic, kws) in enumerate(self.__keywords.items())]
        kw_df = pd.DataFrame(kws_per_topic, columns=["i", "topic", "nr_kws"])
        kw_df["end_idx"] = np.cumsum(kw_df["nr_kws"])
        kw_df["begin_idx"] = kw_df["end_idx"].shift(1).fillna(0).astype(int)

        # Select topic line
        start = kw_df.loc[kw_df['topic'] == topic]["begin_idx"].iloc[0]
        end = kw_df.loc[kw_df['topic'] == topic]["end_idx"].iloc[0]

        # Slice parameter
        q_beta_tilde_shape_topic = self.variational_parameter["a_beta_tilde_S"][start:end]
        q_beta_tilde_rate_topic = self.variational_parameter["b_beta_tilde_S"][start:end]

        # Plot distribution
        topic_kws = list(self.__keywords[topic])
        moments = zip(q_beta_tilde_shape_topic, q_beta_tilde_rate_topic, topic_kws)

        # plot data
        x = np.linspace(0, x_max, 1000)
        fig, ax = plt.subplots(figsize=(12, 6))
        for shape, rate, keyword in moments:
            # ax.fill(x, stats.gamma.pdf(x, a = shape, scale = 1/rate), alpha = .5)
            if detail == False:
                ax.plot(x, stats.gamma.pdf(x, a=shape, scale=1 / rate), alpha=.7, label=f"{keyword}")
            else:
                ax.plot(x, stats.gamma.pdf(x, a=shape, scale=1 / rate), alpha=.7,
                        label=r"$q({}){},{}$ = Gamma({:.2f}, {:.2f})".format(r"\tilde{\beta}", topic, keyword, shape,
                                                                             rate))

        ax = plt.gca()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(2.5)
            ax.spines[axis].set_color("0.2")

        plt.xlabel('Value', fontsize=13, color="0.2")
        plt.ylabel('Probability Density', fontsize=15, color="0.2")
        plt.title(
            'Variational adjusted topic-word distribution \n' r'for topic {}: $q({}){}$'.format(
                topic, r"\tilde{\beta}",topic), fontsize=15, weight="bold", color="0.2")

        plt.legend(loc=1, frameon=False, labelcolor="0.2",
                   prop={"weight": "bold", "size": 13})
        plt.grid(alpha=.3)
        plt.show()

    def plot_word_distribution(self, word: str, topic: str, x_max: int =10):
        """
        Shows the fitted variational distribution of q(\Tilde{\beta}){topic,word} and q(\beta^*)_{topic,word}.
        :param word: Word for which the distribution should be plotted
        :param topic: Topic for the topic-word distribution
        :param x_max: Maximum x value for the plot
        """
        # Check if topic str is valid
        topic_names = list(self.keywords.keys())
        rs_names = [f"residual_topic_{i + 1}" for i in range(self.residual_topics)]
        if topic not in topic_names + rs_names:
            raise ValueError(f"{topic} is not a valid topic name. Topic names are: {topic_names + rs_names}")

        # Check if word exists in the vocabulary
        if word not in list(self.model_settings["vocab"]):
            raise ValueError(f"{word} not in vocabulary!")

        def search_idx(word="abc", topic="abc"):
            """
            Returns the index of the keyword in the keyword dictionary - which is later 1 dimensional in beta tilde.
            There are 2 cases to be considered here:
            (1) A keyword is only used once
            (2) A keyword is used multiple times in multiple topics
            """

            kws_per_topic = []
            for idx, key in enumerate(self.__keywords.keys()):
                kws_per_topic.append(len(self.__keywords[key]))

            for key, value in self.__keywords.items():
                if key == topic:
                    if word in value:
                        word_in_topic_index = list(value).index(word)
                        word_in_topic_index
                    else:
                        print(f"{word} not defined as keyword for topic {topic}")
                        return None

            if list(self.__keywords.keys()).index(topic) != 0:
                return np.sum(kws_per_topic[:list(self.__keywords.keys()).index(topic)]) + word_in_topic_index
            else:
                return word_in_topic_index

        # Return q(\tilde[\beta}) parameter if the word is a seed word
        if word in list(self.__keywords[topic]):
            keywords_all = [kw for kws in self.__keywords.values() for kw in kws]
            kw_index = search_idx(word=word, topic=topic)

            beta_tilde_shape = self.variational_parameter["a_beta_tilde_S"][kw_index]
            beta_tilde_rate = self.variational_parameter["b_beta_tilde_S"][kw_index]
        else:
            print(f"{word} is not defined as a keyword for topic {topic}!")

        # beta word index
        word_idx = list(self.model_settings["vocab"]).index(word)
        topic_idx = list(self.__keywords.keys()).index(topic)

        # get beta moments
        beta_shape = self.variational_parameter["a_beta_S"][(topic_idx, word_idx)]
        beta_rate = self.variational_parameter["b_beta_S"][(topic_idx, word_idx)]

        # zip data
        if word in list(self.__keywords[topic]):
            # compute combined
            moments = zip([beta_shape, beta_tilde_shape], [beta_rate, beta_tilde_rate], [r"\beta^*", r"\tilde{\beta}"])
        else:
            moments = zip([beta_shape], [beta_rate], ["beta"])

        # plot data
        x = np.linspace(0, x_max, 1000)
        fig, ax = plt.subplots(figsize=(12, 6))
        for shape, rate, betas in moments:
            # ax.fill(x, stats.gamma.pdf(x, a = shape, scale = 1/rate), alpha = .5)
            ax.plot(x, stats.gamma.pdf(x, a=shape, scale=1 / rate), alpha=.7,
                    label=r"$q({})_({},{})$ = Gamma({:.2f}, {:.2f})".format(
                        betas, topic, word, shape, rate))

        ax = plt.gca()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(2.5)
            ax.spines[axis].set_color("0.2")

        plt.xlabel('Value', fontsize=13, color="0.2")
        plt.ylabel('Probability Density', fontsize=15, color="0.2")
        plt.title(f'Variational topic-word distributions', fontsize=15, weight="bold", color="0.2")

        plt.legend(loc=1, frameon=False, labelcolor="0.2",
                   prop={"weight": "bold", "size": 13})
        plt.grid(alpha=.3)
        plt.show()

    def __repr__(self):
        return f"Seeded Poisson Factorization (seededpf) model initialized with {len(self.keywords.keys())} keyword " \
               f"topics and {self.residual_topics} residual topics."

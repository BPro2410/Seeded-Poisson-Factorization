import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.feature_extraction.text import CountVectorizer

from SPF.SPF_helper import SPF_helper, SPF_lr_schedules

# Shortcuts
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfp.distributions.JointDistributionCoroutine.Root



class SPF(tf.keras.Model):
    """
    Tensorflow implementation of Variational Seeded Topic Model.
    """

    def __init__(self, keywords: dict,
                 residual_topics: int = 0):
        """

        :param keywords: Dictionary containing topics (keys) and keywords (values) of the form
        {'topic_1':['word1', 'word2'],'topic_2':['word1', 'word2']}
        :param residual_topoics: Number of residual topics (i.e. topics with no prior information available)
        """

        super(SPF, self).__init__()
        # Initialize model parameters
        # self.keywords = SPF_helper._check_keywords(keywords)
        self.keywords = keywords
        self.residual_topics = residual_topics
        self.model_settings = {"num_topics" : len(self.keywords.keys()) + residual_topics}
        # self.num_topics = len(self.keywords)


    def read_docs(self,
                  text: list[str],
                  count_vectorizer = CountVectorizer(stop_words = "english", min_df = 2),
                  batch_size: int = 1024,
                  seed: int = 2410):
        """
        Reads documents, processes them into the form required by the SPF model and creates additional metadata.

        :param text: Text to be classified.
        :param count_vectorizer: CountVectorizer object used to create the DTM.
        :param batch_size: Batch_size used for training.
        :param seed: Seed used for shuffeling the data.
        :return: None
        """

        # Initialize vectorizer
        cv = count_vectorizer
        cv.fit(text)

        # Create DTM
        counts = sparse.csr_matrix(cv.transform(text), dtype = np.float32)

        # check if there are documents with 0 tokens
        # doc_lengths = tf.reduce_sum(counts.toarray(), axis = 1)
        # if tf.sort(doc_lengths)[0] < 1:
        #     raise ValueError("Please review documents or provide a custom tokenizer. There are documents with 0 tokens.")
        zero_idx = np.where(np.sum(counts, axis = 1) == 0)
        if len(zero_idx[0]) > 0:
            raise ValueError(f"There are documents with zero words after tokenization. "
                             f"Please remove them or provide a custom tokenizer. Documents: {zero_idx[0]}")

        # Add metadata
        self.model_settings["vocab"] = cv.get_feature_names_out()
        self.model_settings["num_words"] = len(self.model_settings["vocab"])
        self.model_settings["num_documents"] = counts.shape[0]
        doc_lengths = tf.reduce_sum(counts.toarray(), axis=1)
        self.model_settings["doc_length_K"] = tf.concat([doc_lengths[:, tf.newaxis]] *
                                                        self.model_settings["num_topics"], axis = 1)


        # Check if seed words are contained in vocabulary & create keyword indices tensor for beta-tilde
        self.kw_indices_topics = list()
        not_in_vocab_words = list()

        for idx, topic in enumerate(self.keywords.keys()):
            # idx indicates the topic index -> used for beta-tilde adjustments
            for keyword in self.keywords[topic]:
                try:
                    kw_index = list(self.model_settings["vocab"]).index(keyword)
                    self.kw_indices_topics.append([idx, kw_index])
                except Exception:
                    print(f"### Keyword: {keyword} from topic {topic} is not in vocabulary. "
                          f"Keyword dictionary will be pruned.")
                    not_in_vocab_words.append(keyword)

        # Remove keywords which are not in vocab!
        for topic in self.keywords.keys():
            for na_word in not_in_vocab_words:
                try:
                    self.keywords[topic].remove(na_word)
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
        self.num_kw = len([kw for kws in self.keywords.values() for kw in kws])
        self.prior_parameter["a_beta_tilde"] = tf.fill(dims = [self.num_kw], value = self.prior_params["beta_tilde_shape"])
        self.prior_parameter["b_beta_tilde"] = tf.fill(dims = [self.num_kw], value = self.prior_params["beta_tilde_rate"])

        # --- Create free variational family parameters ---
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
        def variational_family():
            theta_surrogate = yield tfd.Gamma(
                self.variational_parameter["a_theta_S"], self.variational_parameter["b_theta_S"],
                name="theta_surrogate")
            beta_surrogate = yield tfd.Gamma(
                self.variational_parameter["a_beta_S"], self.variational_parameter["b_beta_S"],
                name="beta_surrogate")
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
                             kw_indices, document_indices, doc_lengths):
            """
            keyATM as GaP model!
            """

            # Theta over documents
            theta = yield tfd.Gamma(a_theta, b_theta, name="document_topic_distribution")

            # Beta over Topics
            beta = yield tfd.Gamma(a_beta, b_beta, name="topic_word_distribution")
            beta_tilde = yield tfd.Gamma(a_beta_tilde, b_beta_tilde, name="adjusted_topic_word_distribution")
            beta_star = tf.tensor_scatter_nd_add(beta, kw_indices, beta_tilde)

            # Scale theta by document length (according to GaP paper)
            theta_batch = tf.gather(theta, document_indices)
            relevant_doc_lengths = tf.gather(doc_lengths, document_indices)
            theta_bar = theta_batch / relevant_doc_lengths

            # Word counts
            rate = tf.matmul(theta_bar, beta_star)
            y = yield tfd.Poisson(rate=rate, name="word_count")

        model_joint = tfp.distributions.JointDistributionCoroutineAutoBatched(
            lambda: generative_model(self.prior_parameter["a_theta"], self.prior_parameter["b_theta"],
                                     self.prior_parameter["a_beta"], self.prior_parameter["b_beta"],
                                     self.prior_parameter["a_beta_tilde"], self.prior_parameter["b_beta_tilde"],
                                     self.kw_indices_topics, document_indices, self.model_settings["doc_length_K"])
        )

        return model_joint


    def __model_joint_log_prob(self, theta, beta, beta_tilde, document_indices, counts):
        """
        Prior loss function.
        :param theta: Theta surrogate sample.
        :param beta: Beta surrogate sample.
        :param beta_tilde: Beta tilde surrogate sample.
        :param document_indices: Document indices relevant for batches
        :param counts: DTM counts (batched).
        :return: Model prior loss.
        """
        model_joint = self.__create_prior_batched(document_indices)
        return model_joint.log_prob_parts([theta, beta, beta_tilde, counts])



    def model_train(self, lr: float = 0.1,
                    epochs: int = 500,
                    lr_scheduler: str = "dynamic",
                    lr_scheduler_params = {"check_each": 150, "check_last":50, "threshold":0.001},
                    tensorboard: bool = False,
                    log_dir: str = os.getcwd(),
                    save_every: int = 1,
                    priors: dict = {},
                    variational_parameter: dict = {}):
        """
        Model training.
        :param lr: Learning rate for Adam optimizer.
        :param epochs: Model iterations.
        :param lr_scheduler: Indicates if a learning rate scheduler should be used. Use 'None' for no scheduling.
        :param lr_scheduler_params: Parameter for the learning rate scheduler.
        :param tensorboard: Indicator whether tensorflow logs should be saved.
        :param log_dir: Directory for the tensorboard logs.
        :param save_every: Tensorboard log interval.
        :param priors: Dictionary containing the prior parameter.
        :param variational_parameter: Dictionary containing the initial variational_parameter.
        :return: None
        """

        # Check prior parameter and variational parameter values
        self.prior_params = SPF_helper._check_priors(priors)
        self.variational_params = SPF_helper._check_variational_parameter(
            variational_parameter,
            corpus_info = self.model_settings["num_documents"])

        # Create all the required model parameters in matching dimensions
        self.__create_model_parameter()

        # Define the variational family
        self.variational_family = self.__create_variational_family()

        # Initialize model parameters and storage matrices
        optim = tf.optimizers.Adam(learning_rate = lr)
        neg_elbos = list()
        entropys = list()
        log_priors = list()
        recon_losses = list()

        def progress_bar(progress, total, epoch_runtime=0):
            """
            Simple progress bar to visualize the model runtime.
            """
            percent = 100 * (progress / float(total))
            bar = "*" * int(percent) + "-" * (100 - int(percent))
            print(f"\r|{bar}| {percent:.2f}% [{epoch_runtime:.4f}/s per epoch]", end="\r")

        @tf.function
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
                # Sample from the variational family
                theta, beta, beta_tilde = self.variational_family.sample()

                # Compute log prior loss
                log_prior_losses = self.__model_joint_log_prob(
                    theta, beta, beta_tilde, document_indices, tf.sparse.to_dense(outputs)
                )
                log_prior_loss_theta, log_prior_loss_beta, \
                    log_prior_loss_beta_tilde, reconstruction_loss = log_prior_losses

                # Rescale reconstruction loss since it is only based on a mini-batch
                recon_scaled = reconstruction_loss * tf.dtypes.cast(
                    tf.constant(self.model_settings["num_documents"]) / (tf.shape(outputs)[0]),
                                                                    tf.float32)
                log_prior = tf.reduce_sum(
                    [log_prior_loss_theta, log_prior_loss_beta, log_prior_loss_beta_tilde, recon_scaled])

                # Compute entropy loss
                entropy = self.variational_family.log_prob(theta, beta, beta_tilde)

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
        progress_bar(0, epochs)
        self.lrs = [lr]

        # Start iterating
        for idx, epoch in enumerate(range(epochs)):
            # Capture epoch metrics
            start_time = time.time()
            epoch_loss = list()
            epoch_entropy = list()
            epoch_log_prior = list()
            epoch_reconstruction_loss = list()

            # Check if lr should be changed
            if (lr_scheduler == "dynamic") and (epoch % lr_scheduler_params["check_each"] == 0):
                optim.lr.assign(SPF_lr_schedules.dynamic_schedule(epoch = epoch,
                                                                  optim = optim,
                                                                  losses = neg_elbos,
                                                                  check_each=lr_scheduler_params["check_each"],
                                                                  check_last=lr_scheduler_params["check_last"],
                                                                  threshold = lr_scheduler_params["threshold"]))
            self.lrs.append(optim.lr.numpy())

            # Iterate through batches
            for batch_index, batch in enumerate(iter(self.dataset)):
                batches_per_epoch = len(self.dataset)
                step = batches_per_epoch * epoch + batch_index

                inputs, outputs = batch  # inputs = {'document_indices':[]}, outputs = counts

                # Calculate loss
                neg_elbo, entropy, prior_loss, recon_loss = train_step(inputs, outputs, optim)

                # Store batch loss in epoch loss
                epoch_loss.append(neg_elbo)
                epoch_entropy.append(entropy)
                epoch_log_prior.append(prior_loss)
                epoch_reconstruction_loss.append(recon_loss)

            # Store end of batch metrics
            end_time = time.time()
            neg_elbos.append(np.mean(epoch_loss))
            entropys.append(np.mean(epoch_entropy))
            log_priors.append(np.mean(epoch_log_prior))
            recon_losses.append(np.mean(epoch_reconstruction_loss))

            if tensorboard == True and epoch % save_every == 0:
                tf.summary.text("topics", self.print_topics(), step = epoch)
                tf.summary.scalar("elbo/Negative ELBO", neg_elbos[-1], step = epoch)
                tf.summary.scalar("elbo/Entropy loss", entropys[-1], step = epoch)
                tf.summary.scalar("elbo/Log prior loss", log_priors[-1], step=epoch)
                tf.summary.scalar("elbo/Reconstruction loss", recon_losses[-1], step=epoch)
                tf.summary.histogram("params/Theta shape surrogate",
                                     self.variational_parameter["a_theta_S"], step = epoch)
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

            progress_bar(idx + 1, epochs, end_time - start_time)

        # Store model loss as object attribute
        self.model_loss = {"negative_elbo": np.array(neg_elbos), "entropy": np.array(entropys),
                           "log_prior": np.array(log_priors), "reconstruction_loss": np.array(recon_losses)}

    def calculate_topics(self):
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
                return "No_keyword_topic_" + str(i)

        topics = [recode_cats(i) for i in categories]
        return topics, E_theta

    def plot_model_loss(self,
                        neg_elbo: bool = True,
                        detail_plot: bool = False,
                        neg_elbo_lr_changes: bool = False):
        """
        Plot model loss to check convergence.
        :param neg_elbo: Indicator whether the negative ELBO loss should be plotted.
        :param detail_plot: Indicator whether all losses should be plotted.
        :param neg_elbo_lr_changes: Indicator whether the learning rate changes should be plotted on the
        negative ELBO loss plot.
        :return: None
        """
        if neg_elbo == True:
            fig, ax1 = plt.subplots(figsize=(12, 7))
            plt.title(f"SPF loss plot on {self.model_settings['num_documents']} documents",
                      fontsize = 15, weight = "bold", color = "0.2")
            ax1.set_xlabel("Epoch", fontsize = 13, color = "0.2")
            ax1.set_ylabel("Negative ELBO loss", fontsize = 15, color = "0.2")
            lns1 = ax1.plot(self.model_loss["negative_elbo"], color = "black", label = "Negative ELBO", lw = 2.5, mec = "w",
                     mew = "2", alpha = 0.9)
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



        if detail_plot == True:
            show_since = 0
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 11))
            axs[0].set_title("Log Likelihood loss with fixed data")
            axs[0].plot(self.model_loss["log_prior"][show_since:], label="log_prior_loss")
            axs[0].plot(self.model_loss["reconstruction_loss"][show_since:], label="reconstruction_loss")
            axs[0].set_ylabel("Log Likelihood")
            axs[0].legend()
            axs[0].grid(True)
            axs[1].set_title("Entropy loss")
            axs[1].plot(self.model_loss["entropy"][show_since:])
            axs[1].set_ylabel("Entropy")
            axs[1].grid(True)
            axs[2].set_title("Negative ELBO loss")
            axs[2].plot(self.model_loss["negative_elbo"][show_since:])
            axs[2].set_ylabel("Neg ELBO")
            axs[2].grid(True)
            axs[2].set_xlabel("Iteration")
            fig.suptitle("Variational Inference model losses")
            plt.show()

        # if neg_elbo_lr_changes == True:
        #     unique_lrs = np.unique(self.lrs)[::-1]
        #     lr_change_at_epoch = np.array([self.lrs.index(i) for i in unique_lrs])
        #     unique_lrs = unique_lrs[1:]
        #     lr_change_at_epoch = lr_change_at_epoch[1:]
        #
        #     plt.figure(figsize=(12, 7))
        #     plt.title("Negative ELBO loss and learning rate changes")
        #     plt.plot(self.model_loss["negative_elbo"])
        #     for idx, i in enumerate(unique_lrs):
        #         plt.annotate(f"LR: {i:.4f}",
        #                      xy=(lr_change_at_epoch[idx], plt.gca().get_ylim()[0]),
        #                      xytext=(lr_change_at_epoch[idx], plt.gca().get_ylim()[0] - 0.15),
        #                      arrowprops=dict(facecolor="black", shrink=0.05, alpha=0.3))
        #     plt.grid(True, alpha=0.3)
        #     plt.show()


    def calculate_topic_word_distributions(self):
        """
        Calculate posterior means for the topic-word distribution.
        :return: Topic-word distribution dataframe.
        """
        E_beta = self.variational_parameter["a_beta_S"] / self.variational_parameter["b_beta_S"]
        E_beta_tilde = self.variational_parameter["a_beta_tilde_S"] / self.variational_parameter["b_beta_tilde_S"]
        beta_star = tf.tensor_scatter_nd_add(E_beta, self.kw_indices_topics, E_beta_tilde)
        topic_names = list(self.keywords.keys())
        rs_names = [f"residual_topic_{i}" for i in range(self.residual_topics)]
        return pd.DataFrame(tf.transpose(beta_star), index=self.model_settings["vocab"], columns=topic_names+rs_names)

    def print_topics(self):
        E_beta = self.variational_parameter["a_beta_S"] / self.variational_parameter["b_beta_S"]
        E_beta_tilde = self.variational_parameter["a_beta_tilde_S"] / self.variational_parameter["b_beta_tilde_S"]
        beta_star = tf.tensor_scatter_nd_add(E_beta, self.kw_indices_topics, E_beta_tilde)
        top_words = np.argsort(-beta_star, axis=1)
        topic_strings = list()
        for topic_idx in range(self.model_settings["num_topics"]):
            neutral_start_string = "{}:".format(list(self.keywords.keys())[topic_idx])
            words_per_topic = 50
            neutral_row = [self.model_settings["vocab"][word] for word in top_words[topic_idx, :words_per_topic]]
            neutral_row_string = ", ".join(neutral_row)
            neutral_string = " ".join([neutral_start_string, neutral_row_string])

            topic_strings.append(" \n".join(
                [neutral_string]
            ))
        return np.array(topic_strings)

    def __repr__(self):
        return f"Seeded Poisson Factorization (SPF) model initialized with {len(self.keywords.keys())} keyword " \
               f"topics and {self.residual_topics} residual topics."


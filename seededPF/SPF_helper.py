import numpy as np


class SPF_helper:

    def __init__(self):
        pass

    @staticmethod
    def _check_keywords(keywords, residual_topics = 0):
        """
        Helper function that checks if the keywords passed are correctly.
        """

        # Check correct data type
        if not isinstance(keywords, dict):
            raise TypeError("Keywords should be passed as a dictionary, e.g. {'topic_1':['word1', 'word2'], "
                            "'topic_2':['word1', 'word2']}")

        # Check length of keywords
        if len(keywords) == 0:
            # raise ValueError("Please provide keywords dictionary with topics, e.g. {'topic_1':['word1', 'word2'], "
            #                  "'topic_2':['word1', 'word2']}")
            if residual_topics <= 0:
                raise ValueError("If no keywords are provided, the number of residual topics have to be specified.")
            print(f"No keywords provided. A standard topic model with {residual_topics} residual topics will be fitted.")

        # Check format
        for topic, kws in keywords.items():
            for keyword in kws:
                if not isinstance(keyword, str):
                    raise TypeError(f"Keyword: {keyword} in topic {topic} should be a string.")

        return keywords

    @staticmethod
    def _check_priors(priors):
        """
        Helper function that checks if the prior parameter are passed correctly. If not, the standard
        configuration is used.
        """

        prior_names = ["theta_shape", "theta_rate", "beta_shape", "beta_rate", "beta_tilde_shape", "beta_tilde_rate"]

        if not isinstance(priors, dict):
            raise TypeError(f"Prior parameter should be passed as a dictionary with prior parameter names "
                            f"possible: {priors}")

        # Check prior specification
        for prior, value in priors.items():
            if prior not in prior_names:
                raise ValueError(f"Please specify correct prior parameters! Otherwise use standard specification! "
                                 f"Prior parameter possible: {priors}")
            if type(value) != float:
                raise ValueError("Prior values must be floating values!")

        # Check if all prior are specified
        priors["theta_shape"] = priors.get("theta_shape", 0.3)
        priors["theta_rate"] = priors.get("theta_rate", 0.3)
        priors["beta_shape"] = priors.get("beta_shape", 0.3)
        priors["beta_rate"] = priors.get("beta_rate", 0.3)
        priors["beta_tilde_shape"] = priors.get("beta_tilde_shape", 1.0)
        priors["beta_tilde_rate"] = priors.get("beta_tilde_rate", 0.3)

        return priors

    @staticmethod
    def _check_variational_parameter(variational_parameter, corpus_info):
        """
        Helper function that checks if the variational parameter are passed properly. If not, the standard
        configuration is used.
        """

        vp_names = ["theta_shape_S", "theta_rate_S", "beta_shape_S", "beta_rate_S",
                    "beta_tilde_shape_S", "beta_tilde_rate_S"]

        if not isinstance(variational_parameter, dict):
            raise TypeError(f"Variational parameter should be passed as a dictionary with variational parameter"
                            f"names possible: {vp_names}")

        # Check prior specification
        for prior, value in variational_parameter.items():
            if prior not in vp_names:
                raise ValueError(f"Please specify correct variational parameters! Otherwise use standard specification!"
                                 f"Variational parameter names possible: {vp_names}")
            if type(value) != float:
                raise ValueError("Variational parameter values must be floating values!")

        # Check if all prior are specified
        variational_parameter["theta_shape_S"] = variational_parameter.get("theta_shape_S", 1.0)
        variational_parameter["theta_rate_S"] = variational_parameter.get("theta_rate_S", corpus_info / 1000)
        variational_parameter["beta_shape_S"] = variational_parameter.get("beta_shape_S", 1.0)
        variational_parameter["beta_rate_S"] = variational_parameter.get("beta_rate_S", corpus_info / 1000 * 2)
        variational_parameter["beta_tilde_shape_S"] = variational_parameter.get("beta_tilde_shape_S", 2.0)
        variational_parameter["beta_tilde_rate_S"] = variational_parameter.get("beta_tilde_rate_S", 1.0)

        return variational_parameter


class SPF_lr_schedules:
    def __init__(self):
        pass

    @staticmethod
    def power_scheduling(epoch, steps=200, initial_lr=0.1, power=1):
        """
        Learning rate as a function of the iteration number t: eta(t) = eta_0 / (1+t/s)^c.
        The steps s, power c and initial learning rate eta_0 are hyperparameters.

        :param epoch: Current iteration
        :param steps: Steps
        :param initial_lr: Initial learning rate
        :param power: Power
        :return: New learning rate.
        """

        return initial_lr / (1 + epoch / steps) ** 1

    @staticmethod
    def piecewise_constant_scheduling(epoch,
                                      epoch_intervall: list[int] = [100, 200, 300],
                                      new_lrs: list[int] = [0.01, 0.001, 0.0001]):
        """
        Use constant learning rates for a given number of epochs.
        :param epoch: Current epoch.
        :param epoch_intervall: Number at which a learning rate change takes place.
        :param new_lrs: New learning rate at epoch threshold.
        :return: New learning rate.
        """

        if epoch == epoch_intervall[0]:
            return new_lrs[0]
        elif epoch == epoch_intervall[1]:
            return new_lrs[1]
        elif epoch == epoch_intervall[2]:
            return new_lrs[2]

    @staticmethod
    def dynamic_schedule(epoch: int,
                         optim,
                         losses: list[float],
                         check_each: int = 150,
                         check_last: int = 50,
                         threshold: float = 0.001):
        """
        If the average percentage change over the last x epochs is smaller than a certain threshold,
        we half the lr in order to improve model training.
        :param epoch: Current epoch.
        :param optim: Optimizer used.
        :param losses: List with last losses.
        :param check_each: Check update condition each x epochs.
        :param check_last: Interval which is checked for the update condition.
        :param threshold: Threshold for the update condition.
        :return: New learning rate
        """

        if epoch % check_each == 0:

            # Get last losses
            last_losses = losses[-check_last:]
            # Compute pct change
            loss_pct_change = np.abs(np.diff(last_losses) / last_losses[:-1])
            mean_loss_pct_change = np.mean(loss_pct_change)

            # Half learning rate
            if epoch > check_each:
                if mean_loss_pct_change < threshold:
                    return optim.lr.numpy() / 2
            return optim.lr.numpy()
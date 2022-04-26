import threading
import numpy as np


def logsumexp(Z):  # dimension C x N
    Zmax = np.max(Z, axis=0)[None, :]  # max over C
    log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=0))
    return log_sum_exp


def evaluate_acc(y_test, y_pred):
    """
    Evaluates the accuracy of a model's prediction
    :param y_test: np.ndarray - the true labels
    :param y_pred: np.ndarray - the predicted labels
    :return: float - prediction accuracy
    """
    return np.sum(y_pred == y_test) / y_pred.shape[0]


class NaiveBayes:
    """
    Threaded implementation of the Naive Bayes ML algorithm using the Multinomial likelihood
    to classify text documents.
    """
    def __init__(self):
        pass

    def fit(self, x, y):
        """
        Fits the model to the given training data by learning the model parameters for each feature & computes the
        prior for each class.
        :param x: np.ndarray - the training data represented as a count matrix using CountVectorizer from sklearn
        :param y: np.ndarray - the labels corresponding to the training data (in the same order). Labels must start at
                               0 and end at (# of labels - 1). e.g. if there are 3 possible labels, they must be 0, 1, 2
        :return: the NaiveBayes model fitted on the given data
        """
        N, D = x.shape
        self.C = np.max(y) + 1  # stores the num of labels/categories
        probs = np.zeros((self.C, D))   # initializes the array that will store the model parameters
        Nc = np.zeros(self.C)  # initializes the array that will store the # of instances of each label/category

        # for each class get the MLE for each d,c (rel frequencies)
        for c in range(self.C):
            x_c = x[y == c]  # filter the rows of the data to all rows where label = c
            Nc[c] = x_c.shape[0]  # stores the number of instances where label = c

            num_threads = 10
            # calculates the interval of features for which each thread will work in
            interval_size = int(D / num_threads)
            threads = []    # stores the created threads
            for thread in range(num_threads):
                # creates threads
                threads.append(
                    threading.Thread(
                        target=NaiveBayes._fit_thread,
                        args=(x_c, c, thread*interval_size, (thread*interval_size) + interval_size, probs))
                )
                threads[thread].start()

            # waits for all threads to finish execution
            for thread in threads:
                thread.join()

            # without threading - this is the work being split up across n threads
            #for d in range(D):
                # count_d = np.sum(x_c[:, d]) + 1     # counts of word d in all documents labelled c
                # tot_count = np.sum(x_c)             # total word count in all documents labelled c
                # probs[c][d] = count_d / tot_count   # MLE for each d,c (rel frequency)

        self.probs = probs  # stores the learnt model parameters in self
        self.pi = (Nc + 1) / (N + self.C)  # stores the learnt priors using Laplace smoothing
        return self

    def _fit_thread(x_c, c, d_start, d_end, probs):
        """
        Thread fitting functionality, not meant to be used elsewhere.
        """
        for d in range(d_start, d_end):
            count_d = np.sum(x_c[:, d]) + 1  # counts of word d in all documents labelled c
            tot_count = np.sum(x_c)  # total word count in all documents labelled c
            probs[c][d] = count_d / tot_count  # MLE for each d,c (rel frequency)

    def predict(self, xt):
        """
        Predicts the labels of the given instances using the learnt parameters
        :param xt: np.ndarray - the test data represented as a count matrix using CountVectorizer from sklearn
        :return: np.ndarray - the posterior probabilities for each instance for each label. To get the model's label
                              prediction, simply take the label with the highest probability value for each instance.
        """
        Nt, D = xt.shape
        # for numerical stability we work in the log domain
        # we add a dimension because this is added to the log-likelihood matrix
        # that assigns a likelihood for each class (C) to each test point, and so it is C x N
        log_prior = np.log(self.pi)[:, None]

        # computes the Multinomial log likelihoods
        log_likelihood = np.zeros((Nt, self.C))
        for i in range(Nt):
            a = np.log(self.probs ** xt[i])
            b = np.sum(a, axis=1)
            log_likelihood[i] = b

        log_posterior = log_prior.T + log_likelihood
        posterior = np.exp(log_posterior - logsumexp(log_posterior))
        return posterior  # dimension N x C

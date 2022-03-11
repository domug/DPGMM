import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
import colorcet as cc
from scipy.special import loggamma
from scipy.stats import multivariate_t, multivariate_normal
from sklearn.decomposition import PCA


class DPGMM_gibbs():
    """
    Collapsed gibbs sampler for Dirichlet Process Gaussian Mixture Model (DPGMM)
        - DP implementation by Chinese Restaurant Process (CRP)
    """

    def __init__(self, data, alpha=0.01, lambda_0=1., nu_0=False, mu_0=False, S_0=False):
        """
        Parameters
            data: (N,D)
            alpha: scalar 
                DP parameter is assumed to be constant for all of the clusters
            lambda_0: scalar, default as 1
            nu_0: scaler, default as dimension + 2 
            mu_0: (1,D) mean vector, default as sample mean
            S_0: (D,D) covariance matrix, default as diagonal matrix
        """
        self.data = np.array(data)
        try:
            self.N = self.data.shape[0]
            self.D = self.data.shape[1]
        except:
            self.data = data[:, np.newaxis]
            self.N = self.data.shape[0]
            self.D = self.data.shape[1]

        # DP parameter
        self.alpha = alpha

        # hyper parameters of Normal Inverse-Wishart dist'n
        self.lambda_0 = lambda_0
        self.nu_0 = nu_0 if nu_0 else self.D + 2
        self.mu_0 = mu_0 if mu_0 else np.mean(self.data, axis=0)
        self.S_0 = S_0 if S_0 else np.eye(self.D)

        self.num_errors = 0  # for checking numerical errors


    ### for parameter updates ###
    def update_lambda(self, y):
        return self.lambda_0 + len(y)

    def update_nu(self, y):
        return self.nu_0 + len(y)

    def update_mu(self, y, lambda_N):
        N = len(y)
        ybar = np.mean(y, axis=0)
        numer = self.lambda_0 * self.mu_0 + N * ybar
        denom = lambda_N
        return numer / denom

    def update_S(self, y, mu_N, lambda_N):
        N = len(y)
        ybar = np.mean(y, axis=0)

        S = np.zeros((self.D, self.D))
        for i in range(N):
            x = y[i, :][np.newaxis, :]  # (1,D) vector
            S += np.matmul(x.T, x)      # (D,D)

        m0 = self.mu_0[np.newaxis, :]   # (1,D)
        muN = mu_N[np.newaxis, :]

        return self.S_0 + S + self.lambda_0*np.matmul(m0.T, m0) - lambda_N*np.matmul(muN.T, muN)
    

    def update_parameters(self, y):
        """
        y: data instances that belong to specific cluster (N_k, D)
        """
        lambda_updated = self.update_lambda(y)
        nu_updated = self.update_nu(y)
        mu_updated = self.update_mu(y, lambda_updated)
        S_updated = self.update_S(y, mu_updated, lambda_updated)

        return lambda_updated, nu_updated, mu_updated, S_updated


    ### Utility functions ###
    def get_likelihood(self, y_i, lambda_updated, nu_updated, mu_updated, S_updated):
        """
        likelihood calculation for specific cluster assignment
        """
        loc = mu_updated
        shape = (lambda_updated + 1) / (lambda_updated * (nu_updated - self.D + 1)) * S_updated  # (D,D) matrix
        df = nu_updated - self.D + 1
        return multivariate_t.pdf(y_i, loc=loc, shape=shape, df=df)

    @staticmethod
    def normalize_prob(weights):
        """
        normalize posterior cluster weights 
        """
        if np.sum(weights) != 0:
            prob_factor = 1 / sum(weights)
            return [prob_factor * p for p in weights]
        else:  # numerical error
            self.num_errors += 1
            return np.ones(len(weights)) * (1 / len(weights))

    def initialize_clusters(self, K):
        """
        Randomly initialize table assignments
        Parameters
            K: number of initial clusters (default as 1)
        """
        initial_tables = np.linspace(0, K-1, K).astype(int)
        C = np.random.choice(initial_tables, size=self.N)  # randomly initialize table assignments
        return C

    def visualization(self, C, plot):
        """
        For visualizing cluster assignments
        """
        if plot:
            if self.D == 1:
                data = self.data.reshape(-1)
                palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(C)))
                sns.displot(x=data, hue=C, palette=palette, bins=30, aspect=3)
                plt.show()
                
            elif self.D == 2:
                palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(C)))
                plt.figure(figsize=(12,7))
                sns.scatterplot(x=self.data[:, 0], y=self.data[:, 1], s=100, hue=C, palette=palette)
                plt.show()

            else:
                palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(C)))
                pca = PCA(n_components=2)
                data_reduced = pca.fit_transform(self.data)
                plt.figure(figsize=(12,7))
                sns.scatterplot(x=data_reduced[:, 0], y=data_reduced[:, 1], s=100, hue=C, palette=palette)
                plt.show()


    def get_posterior_pi(self, y_i, data, C, k, new=False):
        """
        Calculate posterior cluster weight
        """
        ### new cluster ###
        if new:
            lambda_updated, nu_updated, mu_updated, S_updated = self.update_parameters(y_i)
            prior = self.alpha / (self.N + self.alpha  - 1)
            likelihood = self.get_likelihood(y_i, lambda_updated, nu_updated, mu_updated, S_updated)
            return prior * likelihood
        
        ### existing cluster ###
        existing = data[np.where(C == k)]  # data instances of k-th table
        y = np.concatenate((existing, y_i))
        N_customers = len(y)
        lambda_updated, nu_updated, mu_updated, S_updated = self.update_parameters(y)

        prior = (N_customers-1) / (self.N + self.alpha - 1)
        likelihood = self.get_likelihood(y_i, lambda_updated, nu_updated, mu_updated, S_updated)
        return prior * likelihood     


    ### Sampler ###
    def fit_transform(self, K=1, num_iter=100, verbose=20, plot=True):
        """
        Run collapsed gibbs sampler and update cluster assignments
        Parameters
            K: number of initial clusters (default as 1)
            num_iter: number of sampler iterations
            verbose: to check cluster updates
            plot: 

        """
        # randomly initialize table assignments
        C = self.initialize_clusters(K)

        # run sampler
        start = time.time()
        start1 = time.time()
        for _ in range(num_iter+1):

            # for every predetermined number of iterations, check run time and cluster assignments
            if _ % verbose == 0:
                end = time.time()
                print("########################################################################################")
                print("########################################################################################")
                print("<{} / {}>".format(_, num_iter))
                print(f"{end - start1:.5f} sec\n")
                print(pd.Series(C).value_counts())
                print("\nNumber of numerical errors: {}\n".format(self.num_errors))
                start1 = time.time()
                if plot:
                    self.visualization(C, plot=True)
                    print()

            # iterate over each data instances
            for i in range(self.N): 
                y_i = self.data[i, :][np.newaxis, :]  # convert to (1, D)

                # remove the old table assignment for ith customer
                C_i_rm = np.delete(C, i)
                data_rm = np.delete(self.data, i, axis=0)
                tables = np.unique(C_i_rm) 

                # iterate over each tables and calculate assignment probabilities for existing tables
                posterior_pi = np.empty(len(tables)+1)  # 1 added for new table
                iter = 0
                for k in tables:
                    # update the parameters and calculate posterior clster weight
                    posterior_pi[iter] = self.get_posterior_pi(y_i, data_rm, C_i_rm, k, new=False)
                    iter += 1

                # assignment probability for a new table
                posterior_pi[-1] = self.get_posterior_pi(y_i, y_i, C_i_rm, k, new=True)

                # Assign table for the ith customer
                probabilities = self.normalize_prob(posterior_pi)
                if np.sum(np.isnan(probabilities)):  # for numerical errors
                    self.num_errors += 1
                    continue

                # sample a new table assignment for i-th customer
                choice = np.random.choice(np.append(tables, tables.max()+1), p=probabilities, size=1)
                C[i] = choice

        print("total run time: {:0.5f} sec".format(end-start))
        return C
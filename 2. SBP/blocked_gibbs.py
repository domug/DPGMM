import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from scipy.stats import multivariate_normal, multivariate_t, invwishart
from sklearn.decomposition import PCA
import arviz as az
import time
from tqdm import tqdm_notebook

az.style.use("arviz-darkgrid")

class DPGMM_SBP():
    """
    Blocked gibbs sampler for Dirichlet Process Gaussian Mixture Model (DPGMM)
        - DP implementation w/ stick-breaking process (SBP)
    """

    def __init__(self, data, K=20, alpha=1, lambda_0=1., nu_0=False, mu_0=False, S_0=False):
        """
        Parameters
            data: (N,D)
            K: truncation parameter, default as 25
            alpha, beta: scalar 
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
        self.K = K
        self.alpha = alpha

        # hyper parameters of Normal Inverse-Wishart dist'n
        self.lambda_0 = lambda_0
        self.nu_0 = nu_0 if nu_0 else self.D + 2
        self.mu_0 = mu_0 if mu_0 else np.mean(self.data, axis=0)
        self.S_0 = S_0 if S_0 else np.eye(self.D)

        self.num_errors = 0  # for checking numerical errors


    ### parameter updates ###
    def update_lambda(self, y):
        return self.lambda_0 + len(y)

    def update_nu(self, y):
        return self.nu_0 + len(y)

    def update_mu(self, y, lambda_N):
        ybar = np.mean(y, axis=0)
        numer = (self.lambda_0 * self.mu_0) + (len(y) * ybar)
        denom = lambda_N
        return numer / denom

    def update_S(self, y):
        N = len(y)
        ybar = np.mean(y, axis=0)
        C = np.zeros((self.D, self.D))
        for i in range(N):
            resid = (y[i, :] - ybar)[:, np.newaxis]  # (D,1) vector
            C += np.matmul(resid, resid.T)      # (D,D)
        coef = (self.lambda_0 * N) / (self.lambda_0 + N)
        comp = (ybar - self.mu_0)[:, np.newaxis]
        return self.S_0 + C + coef * np.matmul(comp, comp.T)

    def update_parameters(self, y):
        """
        y: data instances that belong to specific cluster (N_k, D)
        """
        lambda_updated = self.update_lambda(y)
        nu_updated = self.update_nu(y)
        mu_updated = self.update_mu(y, lambda_updated)
        S_updated = self.update_S(y)
        return lambda_updated, nu_updated, mu_updated, S_updated


    ### auxiliary functions ###
    def initialize_parameters(self):
        self.LAMBDAs = np.ones(self.K)
        self.NUs = np.ones(self.K) * self.nu_0
        self.MUs = np.array([self.mu_0 for _ in range(self.K)]) 
        self.Ss = np.array([self.S_0 for _ in range(self.K)]) 
        self.PIs = np.ones(self.K) / self.K
        self.clusters = np.linspace(0, self.K-1, self.K).astype(int)
        self.C = np.random.choice(self.clusters, size=self.N)  # randomly initialize cluster assignments

    def get_likelihood(self, y_i, lambda_updated, nu_updated, mu_updated, S_updated):
        """
        likelihood calculation for specific cluster assignment
        """
        loc = mu_updated
        shape = (lambda_updated + 1) / (lambda_updated * (nu_updated - self.D + 1)) * S_updated  # (D,D) matrix
        df = nu_updated - self.D + 1
        return multivariate_t.pdf(y_i, loc=loc, shape=shape, df=df)

    def stick_breaking_update(self, stick, C):
        weights = np.array([np.random.beta(1+np.sum(C==idx), self.alpha+np.sum(C>idx), size=1) for idx in range(len(stick))]).reshape(-1)
        weights[1:] *= np.cumprod(1 - weights[:-1]) 
        return weights

    @staticmethod
    def normalize_prob(weights):
        """
        normalize posterior cluster weights 
        """
        if np.sum(weights) != 0:
            prob_factor = 1 / sum(weights)
            return [prob_factor * p for p in weights]
        else:  # numerical error
            print("error")
            return np.ones(len(weights)) * (1 / len(weights))

    def visualization(self, C, plot):
        """
        For visualizing cluster assignment results
        """
        if plot:
            if self.D == 1:   # 1-dimensional -> histogram
                data = self.data.reshape(-1)
                palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(C)))
                sns.displot(x=data, hue=C, palette=palette, bins=30, aspect=2)
                plt.show()
                
            elif self.D == 2:  # 2-dimensional -> scatter plot
                palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(C)))
                plt.figure(figsize=(9,4))
                sns.scatterplot(x=self.data[:, 0], y=self.data[:, 1], s=100, hue=C, palette=palette)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                plt.show()

            else:   # 3-dimensional -> scatter plot w/ reduced dimension (PCA)
                palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(C)))
                pca = PCA(n_components=2)
                data_reduced = pca.fit_transform(self.data)
                plt.figure(figsize=(9,4))
                sns.scatterplot(x=data_reduced[:, 0], y=data_reduced[:, 1], s=100, hue=C, palette=palette)
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
                plt.show()

    def block_updates(self):
        # Step 1: update C's by multinomial sampling
        C_temp = np.empty_like(self.C)
        for i in range(self.N):
            y_i = self.data[i, :]
            posterior = np.zeros(self.K)
            for k in range(self.K):
                prior = self.PIs[k]
                likelihood = self.get_likelihood(y_i, self.LAMBDAs[k], self.NUs[k], self.MUs[k], self.Ss[k])
                posterior[k] = prior * likelihood
            posterior_normalized = self.normalize_prob(posterior)
            selected = np.random.choice(self.clusters, p=posterior_normalized, size=1)[0]  # new cluster assignment for ith instance
            C_temp[i] = selected

        # Step 2: update stick-breaking weights (address label switching by weight sorting)
        weights_sorted = pd.Series(self.stick_breaking_update(self.PIs, self.C)).sort_values(ascending=False)
        idx_sorted = weights_sorted.index
        idx_mapping_dict = dict(zip(idx_sorted, self.clusters))
        vfunc = np.vectorize(lambda x: idx_mapping_dict[x])
        self.C = vfunc(C_temp)
        self.PIs = np.array(weights_sorted)

        # Step 3: update parameters
        for k in range(self.K):
            idx = np.where(self.C==k)[0]
            if len(idx) > 0:
                y = self.data[np.where(self.C == k)]  # data instances of k-th table
                self.LAMBDAs[k], self.NUs[k], self.MUs[k], self.Ss[k] = self.update_parameters(y)
            else:
                pass
    

    ### Sampler ###
    def fit_transform(self, num_iter=500, burnin=0.1, verbose=20, plot=True, save=True):
        """
        Blocked Gibbs Sampler for Dirichlet process Gaussian Mixtures
        """
        start = time.time()
        start1 = time.time()
        self.initialize_parameters()
        
        # variables to save traces
        MU = np.array([np.empty_like(self.MUs) for _ in range(num_iter)])
        S = np.array([np.empty_like(self.Ss) for _ in range(num_iter)])
        LAMBDA = np.array([np.empty_like(self.LAMBDAs) for _ in range(num_iter)])
        NU = np.array([np.empty_like(self.NUs) for _ in range(num_iter)])
        PI = np.array([np.empty_like(self.PIs) for _ in range(num_iter)])
        C = np.array([np.empty_like(self.C) for _ in range(num_iter)])


        ### Burn-in period ###
        if burnin:
            num_burnin = int(np.round(num_iter * burnin))
            print("Running burnin period... #samples: {}".format(num_burnin))
            for _ in tqdm_notebook(range(num_burnin)):
                self.block_updates()
            print("Burnin finished. Sampling started...\n")
        else:
            pass


        ### Sampler ###
        for n in tqdm_notebook(range(num_iter)):

            # for every predetermined number of iterations, check run time and cluster assignments
            if (n % verbose == 0) or (n == (num_iter-1)):
                end = time.time()
                print("########################################################################################")
                print("########################################################################################")
                print("<{} / {}>".format(n+1, num_iter))
                print(f"{end - start1:.5f} sec\n")
                print("Posterior cluster weights :\n{}".format(np.round(self.PIs, 3)))
                print("\nNumber of numerical errors: {}\n".format(self.num_errors))
                start1 = time.time()
                if plot:
                    self.visualization(self.C, plot=True)
                    print()

            # collect posterior samples
            self.block_updates()

            # save current step's samples
            C[n] = self.C
            PI[n] = self.PIs
            S[n] = self.Ss
            MU[n] = self.MUs
            LAMBDA[n] = self.LAMBDAs
            NU[n] = self.NUs

        # save sample traces
        traces = {
            "C": C,
            "PI": PI,
            "MU": MU,
            "S": S,
            "NU": NU,
            "LAMBDA": LAMBDA
        }
        end = time.time()

        print("########################################################################################")
        print("########################################################################################")
        print("Sampling finished w/ total run time: {:0.5f} sec".format(end-start))

        # save trace to local directory
        if save:
            temp_dict = {}
            for key in traces.keys():
                temp_dict[key] = traces[key][np.newaxis, :]
            az.to_netcdf(az.dict_to_dataset(temp_dict), "./trace")
            print("Traces of {} saved at current directory.".format(list(traces.keys())))

            self.traces = az.from_netcdf("./trace")
            temp = np.array(self.traces.posterior.PI).mean(axis=1).reshape(-1)
            print("posterior probability greater than 0.05: {}".format(np.where(temp > 0.05)[0]))


    ### posterior inference (TBD) ###
    def plot_trace(self, var_name):
        if var_name not in self.traces.keys():
            print("Invalid variable name.")
        


















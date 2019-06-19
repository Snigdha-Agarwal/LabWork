## https://blog.dominodatalab.com/fitting-gaussian-process-models-python/

import pandas as pd
# import gpmol as gpm
from rdkit import Chem, DataStructs
from numpy.linalg import inv
from rdkit.Chem import AllChem
from sklearn import model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel,RBF,_check_length_scale
from scipy.spatial.distance import pdist,cdist,squareform
from sklearn.preprocessing import StandardScaler

from gaussian_processes_util import plot_gp


def torsionFingerprint(mol):
    return AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)

data = pd.read_csv('calibration_set.csv')
data['Mol'] = data['Smiles'].apply(Chem.MolFromSmiles)
data['Fingerprint'] = data['Mol'].apply(torsionFingerprint)


def GPCalib():
    linear_fit = np.poly1d(np.polyfit(data['Calc_HOMO'], data['Exp_HOMO'], 1))
    linear_fit(data['Calc_HOMO'])



def kernel(X1, l=1.0, sigma_f=1.0):
    print(l)
    ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2. Args: X1: Array of m points (m x d). X2: Array of n points (n x d). Returns: Covariance matrix (m x n). '''
    # sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    nmols = len(X1)
    tanimoto_sim = []
    for i in range(nmols):
        for j in range(i):
            tc = DataStructs.TanimotoSimilarity(X1[i], X1[j])
            tanimoto_sim.append(tc**2)

    # print((-0.5 /l) * np.array(tanimoto_sim))
    K = sigma_f**2 * np.exp(-0.5 / l**2 * np.array(tanimoto_sim))
    K = squareform(K) # covariance matrix should be of dimension 78x78
    np.fill_diagonal(K, 1)
    return K


def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    ''' Computes the suffifient statistics of the GP posterior predictive distribution from m training data X_train and Y_train and n new inputs X_s. Args: X_s: New input locations (n x d). X_train: Training locations (m x d). Y_train: Training targets (m x 1). l: Kernel length parameter. sigma_f: Kernel vertical variation parameter. sigma_y: Noise parameter. Returns: Posterior mean vector (n x d) and covariance matrix (n x n). '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)


    ## Equations below here http://krasserm.github.io/2018/03/19/gaussian-processes/
    # Equation (4) μ∗ = K(T)∗ K(−1)_y y    <-- () superscript _ subscript
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5) Σ∗= K∗∗ − K(T)∗ K(−1)_y K∗
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def fresh():
    X = np.array(list(data['Calc_HOMO']))
    mu = np.zeros(X.shape)
    X_k = data['Fingerprint']
    cov = kernel(X_k)
    # Draw three samples from the prior
    samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

    # Plot GP mean, confidence interval and samples
    plot_gp(mu, cov, X, samples=samples)





def main():
    kernel = RBF_modified() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel)

    X = np.array(list(data['Calc_HOMO']))
    # st = StandardScaler()
    # X = st.fit_transform(X)
    y = data['Exp_HOMO'].values
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    model = gp.fit(X, y)

    scores = model_selection.cross_val_score(model, X, y, cv=20, n_jobs=-1, scoring='neg_mean_absolute_error')
    scores = -1 * scores
    mean_score = scores.mean()
    X_train, X_test, y_train, y_test, idx1, idx2 = model_selection.train_test_split(X, y, np.arange(len(y)),
                                                                                    test_size=1)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    percent_error = np.mean(100 * np.abs(y_test - y_pred_test) / np.abs(y_test))

    print(mean_score,percent_error)


class RBF_modified(RBF):
    def __call__(self, X, Y=None, eval_gradient=False):
        mols = data['Smiles'].apply(Chem.MolFromSmiles)
        X= mols.apply(torsionFingerprint)
        X = np.array(list(X))
        st = StandardScaler()
        X = st.fit_transform(X)
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric='jaccard')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                    / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K


if __name__ == '__main__':
    main()   # using GP library
    # fresh()  # GP from scratch
    # GPCalib() # code by alan-aspuru
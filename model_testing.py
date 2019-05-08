import matplotlib
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ml import BitVect_to_NumpyArray

matplotlib.use("TkAgg")


class trainModels:
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.kernelRidge=self.trainKR()
        self.ridge=self.trainRidge()
        self.gaussian=self.trainGaussian()
        self.randomForest = self.trainRandomForest()

    def trainKR(self):
        KRmodel = GridSearchCV(KernelRidge(), cv=10,
                               param_grid={"alpha": np.logspace(-10, -5, 10),
                                           "gamma": np.logspace(-12, -9, 10), "kernel": ['laplacian', 'rbf']},
                               scoring='neg_mean_absolute_error', n_jobs=-1)

        KRmodel = KRmodel.fit(self.X, self.y)
        Best_KernelRidge = KRmodel.best_estimator_
        print("Best Kernel Ridge model")
        print(KRmodel.best_params_)
        print(-1 * KRmodel.best_score_)
        return Best_KernelRidge

    def trainRidge(self):
        Rmodel = GridSearchCV(Ridge(), cv=20,
                              param_grid={"alpha": np.logspace(-10, -5, 30), }, scoring='neg_mean_absolute_error',
                              n_jobs=-1)

        Rmodel = Rmodel.fit(self.X, self.y)
        Best_Ridge = Rmodel.best_estimator_
        print("Best Ridge model")
        print(Rmodel.best_params_)
        print(-1 * Rmodel.best_score_)
        return Best_Ridge

    def trainGaussian(self):
        GPmodel = GridSearchCV(GaussianProcessRegressor(normalize_y=True), cv=20,
                               param_grid={"alpha": np.logspace(-15, -10, 30), }, scoring='neg_mean_absolute_error',
                               n_jobs=-1)
        GPmodel = GPmodel.fit(self.X, self.y)
        Best_GaussianProcessRegressor = GPmodel.best_estimator_
        print("Best Gaussian Process model")
        print(GPmodel.best_params_)
        print(-1 * GPmodel.best_score_)
        return Best_GaussianProcessRegressor

    def trainRandomForest(self):
        RFmodel = GridSearchCV(RandomForestRegressor(), cv=20,
                               param_grid={"n_estimators": np.linspace(50, 150, 25).astype('int')},
                               scoring='neg_mean_absolute_error', n_jobs=-1)

        RFmodel = RFmodel.fit(self.X, self.y)
        Best_RandomForestRegressor = RFmodel.best_estimator_
        print("Best Random Forest model")
        print(RFmodel.best_params_)
        print(-1 * RFmodel.best_score_)
        return Best_RandomForestRegressor


def test_models(trainedModels, X, y):
    '''
    test a bunch of models and print out a sorted list of CV accuracies
            inputs:
                x: training data features, numpy array or Pandas dataframe
                y: training data labels, numpy array or Pandas dataframe

        '''

    '''  model_dict: a dictionary of the form {name : model()}, where 'name' is a string
                            and 'model()' is a sci-kit-learn model object. '''
    model_dict = {
        'Kernel Ridge Regression': trainedModels.kernelRidge,
        'Ridge Regression': trainedModels.ridge,
        'Guassian Process Regressor': trainedModels.gaussian,
        # 'Support Vector Regression': SVR(),
        # 'KNeighborsRegressor': KNeighborsRegressor(),
        # 'Neural Network': MLPRegressor(alpha=100, max_iter=8000, hidden_layer_sizes=[8, 6], early_stopping=False),
        # 'Gradient Boosted Trees': GradientBoostingRegressor(n_estimators=100),
        # 'Random forest': trainedModels.randomForest
    }

    mean_scores = {}
    percent_errors = {}

    for (name, model) in model_dict.items():
        scores = model_selection.cross_val_score(model, X, y, cv=20, n_jobs=-1, scoring='neg_mean_absolute_error')
        scores = -1 * scores
        mean_score = scores.mean()
        mean_scores[name] = mean_score
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        percent_error = np.mean(100 * np.abs(y_test - y_pred_test) / np.abs(y_pred_test))

        percent_errors[name] = percent_error


    sorted_names = sorted(percent_errors, key=mean_scores.__getitem__, reverse=False)


    t = PrettyTable(['name', '%Test Err','Abs Err in CV'])

    for i in range(len(sorted_names)):
        name = sorted_names[i]
        t.add_row([name, round(percent_errors[name],3), round(mean_scores[name],3)])
    print(t)


def main():
    data = pd.read_csv('ml_data.csv')
    # Add some new columns
    data['Mol'] = data['SMILES'].apply(Chem.MolFromSmiles)

    y = data['LUMO'].values

    data['Fingerprint'] = data['Mol'].apply(estate_fingerprint)
    # data['Fingerprint'] = data['Mol'].apply(torsionFingerprint)
    X = np.array(list(data['Fingerprint']))

    st = StandardScaler()
    X = st.fit_transform(X)
    trainedModels = trainModels(X,y)
    test_models(trainedModels,X,y)

def torsionFingerprint(mol):
    return Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)

def estate_fingerprint(mol):
    return FingerprintMol(mol)[0]

if __name__ == '__main__':
    main()

import glob
import os

import matplotlib
import numpy as np
import pandas as pd
from PyFingerprint.All_Fingerprint import get_fingerprint
from prettytable import PrettyTable
from rdkit import Chem
from operator import itemgetter
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, learning_curve, ShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from Fingerprint_test import BitVect_to_NumpyArray

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


def test_models(data,trainedModels, X, y):
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
        'Support Vector Regression': SVR(),
        # 'KNeighborsRegressor': KNeighborsRegressor(),
        # 'Neural Network': MLPRegressor(alpha=100, max_iter=8000, hidden_layer_sizes=[8, 6], early_stopping=False),
        'Gradient Boosted Trees': GradientBoostingRegressor(n_estimators=100),
        'Random forest': trainedModels.randomForest
    }

    mean_scores = {}
    percent_errors = {}
    bias = {}
    variance = {}
    r2 = {}

    #plotting test_train split vs error
    # plt.clf()
    # plt.ylabel('average mean absolute error in CV ', fontsize=20)
    # plt.xlabel('% data in test', fontsize=20)
    names= []

    # Create the figure window
    fig = plt.figure(figsize=(10, 7))

    k=0
    for (name, model) in model_dict.items():
        scores = model_selection.cross_val_score(model, X, y, cv=20, n_jobs=-1, scoring='neg_mean_absolute_error')
        scores = -1 * scores
        mean_score = scores.mean()
        mean_scores[name] = mean_score

        plot_learning_curve(X, fig, k, model, name, y)
        k = k + 1

        # percent_error = []
        # for i in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]:
        X_train, X_test, y_train, y_test,idx1,idx2 = model_selection.train_test_split(X, y, np.arange(len(y)),test_size=0.1)
        #
        model.fit(X_train, y_train)
        #
        y_pred_test = model.predict(X_test)
        #     ##Printing smiles with their predictions
        # # for i in range(len(idx2)):
        # #     print(data['SMILES'][idx2[i]],y_test[i],y_pred_test[i])
        #
        percent_error = np.mean(100 * np.abs(y_test - y_pred_test) / np.abs(y_test))
        #     percent_error.append(np.mean(np.abs(y_test - y_pred_test)))
        # plt.plot( [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],percent_error,'-')
        # names += [name]
        percent_errors[name] = percent_error
        bias[name] = np.mean((y_test-y_pred_test)**2)
        r2[name] = r2_score(y_test,y_pred_test)
    # plt.legend(names, fontsize=15)
    plt.show()

    sorted_names = sorted(percent_errors, key=mean_scores.__getitem__, reverse=False)


    t = PrettyTable(['name', '%Test Err','Abs Err in CV','Bias','R2'])

    for i in range(len(sorted_names)):
        name = sorted_names[i]
        t.add_row([name, round(percent_errors[name],3), round(mean_scores[name],3), round(bias[name],3),round(r2[name],3)])
    print(t)


def plot_learning_curve(X, fig, k, model, name, y):
    ##plotting test and train curve to ee bias and variance
    rs = ShuffleSplit(n_splits=20, test_size=.25, random_state=0)
    sizes, train_scores, test_scores = learning_curve(model, X, y, cv=rs, n_jobs=-1,
                                                      train_sizes=np.linspace(.1, 1.0, 5), scoring='r2')
    # Find the mean and standard deviation for smoothing
    train_std = np.std(train_scores, axis=1)
    train_mean = np.mean(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    # Subplot the learning curve
    ax = fig.add_subplot(2, 3, k + 1)
    ax.plot(sizes, train_mean, 'o-', color='r', label='Training Score')
    ax.plot(sizes, test_mean, 'o-', color='g', label='Testing Score')
    ax.fill_between(sizes, train_mean - train_std, \
                    train_mean + train_std, alpha=0.15, color='r')
    ax.fill_between(sizes, test_mean - test_std, \
                    test_mean + test_std, alpha=0.15, color='g')
    # Labels
    ax.set_title('model = %s' % (name))
    ax.set_xlabel('Number of Training Points')
    ax.set_ylabel('Score')
    ax.set_xlim([0, X.shape[0] * 0.8])
    ax.set_ylim([-0.05, 1.05])


def charge_featurizer(data,pos):
    chargeList = data['Charges'][pos]
    try:
        maxPositiveCharge = max(chargeList,key=itemgetter(1))[1]
    except:
        print(data['Inchi-Key'][pos])
    maxNegativeCharge = min(chargeList,key=itemgetter(1))[1]
    totalPositiveCharge = sum(x[1] for x in chargeList if x[1] > 0)
    totalNegativeCharge = sum(x[1] for x in chargeList if x[1] < 0)
    absAtomicCharge = sum(abs(x[1]) for x in chargeList)/len(chargeList)
    return [maxPositiveCharge,maxNegativeCharge,totalPositiveCharge,totalNegativeCharge,absAtomicCharge]

def addCharge(data):
    for i in range(len(data['Inchi-Key'])):
        chargeDescriptor = charge_featurizer(data,i)
        data['Fingerprint'][i] = np.concatenate([data['Fingerprint'][i], np.array(chargeDescriptor)])
    return data['Fingerprint']

def make_fingerprint(data):
    # data['Fingerprint'] = data['Mol'].apply(estate_fingerprint)
    # data['Fingerprint'] = data['Mol'].apply(lambda x: GetAvalonFP(x))
    data['Fingerprint'] = data['Mol'].apply(torsionFingerprint)
    data['Fingerprint'] = addCharge(data)
    # data['Fingerprint'] = data['SMILES'].apply(pubChemFP)
    # data['Fingerprint'] = data['Mol'].apply(lambda x: GetErGFingerprint(x))
    # data['Fingerprint'] = data['Mol'].apply(lambda x: MACCSkeys.GenMACCSKeys(x))

    # 3D
    # data['Conformer'] = data['Mol'].apply(Chem.rdmolops.RemoveHs)
    # ## Adding conformer
    # path = '../FileConversion/PDBFiles/'
    # for i in range(len(data['Conformer'])):
    #     mol = data['Conformer'][i]
    #     try:
    #         filenames = glob.glob(path + data['Inchi-Key'][i] + '*_S1_solv.pdb')
    #         data['Conformer'][i] = AllChem.AssignBondOrdersFromTemplate(mol, Chem.MolFromPDBFile(filenames[0]))
    #     except:
    #         try:
    #             filenames = glob.glob(path + data['Inchi-Key'][i] + '*_T1_solv.pdb')
    #             data['Conformer'][i] = AllChem.AssignBondOrdersFromTemplate(mol, Chem.MolFromPDBFile(filenames[0]))
    #         except Exception as e:
    #             print('error' +str(e))
    #             AllChem.EmbedMolecule(data['Conformer'][i])
    # data['Fingerprint'] = data['Conformer'].apply(lambda x: Generate.Gen2DFingerprint(x,Gobbi_Pharm2D.factory,dMat=Chem.Get3DDistanceMatrix(x)))
    return data['Fingerprint']

def populateCharges(data):
    path = "../LogFiles/"

    data['Charges'] = np.empty((len(data), 0)).tolist()
    for i in range(len(data['Inchi-Key'])):
        val = data['Inchi-Key'][i]
        chargeList = []
        for file in glob.glob(path+val+'*.log'):
            with open(path + file) as fp:
                start_store = 0
                for line in fp:
                    if "Mulliken charges" in line and "hydrogens summed into heavy atoms:" in line:
                        start_store = 1
                        break
                if start_store == 1:
                    fp.readline()  # ignoring line with 1 and 2
                    for line in fp:
                        parts = line.split()
                        if parts[0] == 'Electronic':  # stopping point
                            break
                        chargeList.append((parts[1], float(parts[2])))
        data['Charges'][i] = chargeList
    return data['Charges']


def main():
    data = pd.read_csv('forSnigdha.csv')
    # Add some new columns
    data['Mol'] = data['SMILES'].apply(Chem.MolFromSmiles)
    data['Charges'] = populateCharges(data)

    y = data['HOMO'].values
    y = y*27.211
    data['Fingerprint'] = make_fingerprint(data)
    X = np.array(list(data['Fingerprint']))

    st = StandardScaler()
    X = st.fit_transform(X)
    trainedModels = trainModels(X,y)
    test_models(data,trainedModels,X,y)


def pubChemFP(mol):
    fp= get_fingerprint(mol,fp_type='pubchem')
    bitvect = [0] * 881
    for val in fp:
        bitvect[val - 1] = 1
    return np.array(list(bitvect))

def torsionFingerprint(mol):
    return Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)

def estate_fingerprint(mol):
    return FingerprintMol(mol)[0]

if __name__ == '__main__':
    main()

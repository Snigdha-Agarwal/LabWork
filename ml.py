import numpy as np
import pandas as pd
import rdkit
from prettytable import PrettyTable
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect, GetErGFingerprint
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem import Descriptors, AllChem
from sklearn import model_selection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def main():

    # ms = Chem.MolFromSmiles('C1CCC1OCC')
    # v = Torsions.GetTopologicalTorsionFingerprintAsIntVect(ms)
    # AllChem.EmbedMolecule(ms)
    # m=Chem.Get3DDistanceMatrix(ms)
    # factory = Gobbi_Pharm2D.factory
    # factory.GetBitDescription(0)
    # fp1= Generate.Gen2DFingerprint(ms,factory,dMat=m)

    data = pd.read_csv('ml_data.csv')
    # Add some new columns
    data['Mol'] = data['SMILES'].apply(Chem.MolFromSmiles)
    data['Mol'].apply(AllChem.EmbedMolecule)

    fp_list = make_fingerprints(data)


    #Checking for HOMO
    print("Homo predictions")
    y = data['HOMO'].values
    test_fingerprints(fp_list, Ridge(alpha=1e-9), y, verbose=True)
    # scores_vs_size = test_fingerprint_vs_size(y,data,Ridge(alpha=1e-9), verbose=True, makeplots=True)

    # Checking for LUMO values
    print("Lumo predictions")
    y = data['LUMO'].values
    test_fingerprints(fp_list, Ridge(alpha=1e-9), y, verbose=True)


def BitVect_to_NumpyArray(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))


class fingerprint():
    def __init__(self, fp_fun, name):
        self.fp_fun = fp_fun
        self.name = name
        self.x = []

    def apply_fp(self, mols):
        for mol in mols:
            fp = self.fp_fun(mol)
            if isinstance(fp, tuple):
                fp = np.array(list(fp[0]))
            if isinstance(fp, rdkit.DataStructs.cDataStructs.ExplicitBitVect):
                fp = BitVect_to_NumpyArray(fp)
            if isinstance(fp, rdkit.DataStructs.cDataStructs.SparseBitVect):
                fp = np.array(list(fp))
            if isinstance(fp, rdkit.DataStructs.cDataStructs.IntSparseIntVect):
                fp = np.array(list(fp))
            if isinstance(fp, rdkit.DataStructs.cDataStructs.UIntSparseIntVect):
                fp = np.array(list(fp))


            self.x += [fp]

            if (str(type(self.x[0])) != "<class 'numpy.ndarray'>"):
                print("WARNING: type for ", self.name, "is ", type(self.x[0]))


def make_fingerprints(data, length=512, verbose=False):
    fp_list = [
        fingerprint(Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect,
                    "Torsion "),
        fingerprint(lambda x: GetMorganFingerprintAsBitVect(x, 2, nBits=length),
                    "Morgan"),
        fingerprint(FingerprintMol, "Estate (1995)"),
        # fingerprint(lambda x: GetAvalonFP(x, nBits=length),
        #             "Avalon bit based (2006)"),
        # fingerprint(lambda x: np.append(GetAvalonFP(x, nBits=length), Descriptors.MolWt(x)),
        #             "Avalon+mol. weight"),
        # fingerprint(lambda x: GetErGFingerprint(x), "ErG fingerprint (2006)"),
        # fingerprint(lambda x: RDKFingerprint(x, fpSize=length),
        #             "RDKit fingerprint"),
        fingerprint(lambda x: Generate.Gen2DFingerprint(x,Gobbi_Pharm2D.factory,dMat=Chem.Get3DDistanceMatrix(x)),
                    "3D pharmacophore"),

    ]

    for fp in fp_list:
        if (verbose): print("doing", fp.name)
        fp.apply_fp(list(data['Mol']))

    return fp_list


def test_model_cv(model, x, y, cv=20):
    # X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)
    scores = model_selection.cross_val_score(model, x, y, cv=cv, n_jobs=-1,
                                              scoring='neg_mean_absolute_error')

    scores = -1 * scores

    return scores.mean()


def test_fingerprints(fp_list, model, y, verbose=True):
    fingerprint_scores = {}

    for fp in fp_list:
        if verbose: print("doing ", fp.name)
        fingerprint_scores[fp.name] = test_model_cv(model, fp.x, y)

    sorted_names = sorted(fingerprint_scores, key=fingerprint_scores.__getitem__, reverse=False)

    t = PrettyTable(["name","avg abs error in CV (kJ/cc)"])
    for i in range(len(sorted_names)):
        name = sorted_names[i]
        t.add_row([name, round(fingerprint_scores[name],3)])
    print((t))


def test_fingerprint_vs_size(y,data,model, num_sizes_to_test = 20, max_size=2048, cv = 20, verbose=False, makeplots=False):

    fp_list = make_fingerprints(data,length = 10) #test
    num_fp = len(fp_list)

    sizes = np.linspace(100,max_size,num_sizes_to_test)

    scores_vs_size = np.zeros([num_fp, num_sizes_to_test])

    num_fp = 0
    for i in range(num_sizes_to_test):
        if verbose: print(i, ",", end='')
        length = sizes[i]
        fp_list = make_fingerprints(data,length = int(length))
        num_fp = len(fp_list)
        for j in range(num_fp):
            scores_vs_size[j,i] = test_model_cv(model, fp_list[j].x, y)

    if (makeplots):

        plt.clf()
        fig = plt.figure(figsize=(10,10))
        fp_names = []
        for i in range(num_fp):
            plt.plot(sizes, scores_vs_size[i,:],'-')
            fp_names += [fp_list[i].name]
        plt.title('Ridge regression, average CV score vs fingerprint length',fontsize=25)
        plt.ylabel('average mean absolute error in CV ',fontsize=20)
        plt.xlabel('fingerprint length', fontsize=20)
        plt.legend(fp_names,fontsize=15)
        plt.ylim([0,0.3])
        plt.show()

    return scores_vs_size



if __name__ == '__main__':
    main()

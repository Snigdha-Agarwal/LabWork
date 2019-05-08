import codecs
import csv
import io
import json
import os


def main():
    path = "../Molecules/"
    csv_file = open('ml_data.csv','w')
    w = csv.writer(csv_file, delimiter=',')
    w.writerow(('SMILES','HOMO','LUMO'))
    for filename in os.listdir(path):
        with open(path + filename) as json_data:
            print(filename)
            d = json.load(json_data)
            w.writerow((d['smiles'],d['s0']['solv']['energies']['homo'],d['s0']['solv']['energies']['lumo']))




if __name__ == '__main__':
    main()

import codecs
import csv
import io
import json
import os


def main():
    path = "../Molecules/"
    xyzPath = "../FileConversion/XYZFiles/"
    csv_file = open('ml_data.csv','w')


    w = csv.writer(csv_file, delimiter=',')
    w.writerow(('SMILES','HOMO','LUMO','RedoxPotential','Inchi-Key'))
    for filename in os.listdir(path):
        with open(path + filename) as json_data:
            print(filename)
            d = json.load(json_data)
            w.writerow((d['smiles'],d['s0']['solv']['energies']['homo'],d['s0']['solv']['energies']['lumo'],d['properties']['rp'],d['inchi-key']))
            #
            #XYZ generation
            xyz = open(xyzPath+d['inchi-key']+'_S1_solv.xyz', 'w')
            geom = d['s1']['solv']['geom']
            xyz.write(str(len(geom)) + '\n\n')
            for i in geom:
                val = i.split(',')
                xyz.write('%s %s %s %s\n' % (val[0], val[1], val[2], val[3]))

            xyz = open(xyzPath+d['inchi-key']+'_T1_solv.xyz', 'w')
            geom = d['t1']['solv']['geom']
            xyz.write(str(len(geom)) + '\n\n')
            for i in geom:
                val = i.split(',')
                xyz.write('%s %s %s %s\n' % (val[0], val[1], val[2], val[3]))



if __name__ == '__main__':
    main()

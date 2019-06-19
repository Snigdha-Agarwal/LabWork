import os
import cclib
from cclib.method import CSPA, MPA


def main():
    path = "../LogFiles/"
    chargeList = []
    for filename in os.listdir(path):
        with open(path + filename) as fp:
            start_store=0
            for line in fp:
                if "Mulliken charges and spin densities with hydrogens summed into heavy atoms:" in line:
                    start_store=1
                    break
            if start_store == 1:
                line = fp.readline() # ignoring line with 1 and 2
                for line in fp:
                    parts = line.split()
                    if parts[0]=='Electronic': # stopping point
                        break
                    chargeList += (parts[1], parts[2])
            print(chargeList)


if __name__ == '__main__':
    main()

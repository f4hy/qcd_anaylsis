#!/usr/bin/env python2
import argparse
import logging
import pandas as pd
import re
import numpy as np
from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from residualmasses import residual_mass
import glob
import os.path


class uk_data_reader:
    """Read data given by the UK group"""

    def __init__(self, data_directory):

        self.data_directory = data_directory

        self.ensembles = ["C0", "C1", "C2", "F1", "M0", "M1", "M2"]

        self.data = {}

        self.readall()

    def readall(self):

        for e in self.ensembles:
            files = glob.glob(os.path.join(self.data_directory, e) + "/*")
            for f in files:
                if "parameters" in f:
                    self.readparameters()
                else:
                    info, bootstraps = self.readfile(f)
                    e, h, obs, central, error = info
                    key = "{}_{}".format(e[-2:],obs)
                    if h:
                        key = "{}_{}_mh{}".format(e[-2:],obs, h)
                    self.data[key] = bootstraps
        #self.data

    def readparameters(self):
        pass

    def readfile(self, f):
        logging.info("Reading {}".format(f))
        heavymass = None
        with open(f) as infile:
            while True:
                line = infile.readline()
                if "Ensembles" in line:
                    ensemble = line.strip(" #\n")
                    continue
                if "Heavy" in line:
                    heavymass = line.split()[-1].strip()
                    continue
                if "Observable" in line:
                    obs = line.split()[-1].strip()
                    continue
                if "central" in line:
                    central = float(infile.readline().strip())
                    continue
                if "error" in line:
                    error = float(infile.readline().strip())
                    continue
                if "Bootstrap" in line:
                    bootstraps = np.fromstring(infile.readline(), sep=',')*1000
                    continue
                if line == '':
                    logging.debug("read empty line so assume EOF")
                    break

        logging.info("{} {} {}".format(ensemble, heavymass, obs))
        logging.info("{} {}".format(np.mean(bootstraps), np.std(bootstraps)))

        # logging.debug("bootstrap mean: {} reported central {}".format(np.mean(bootstraps), central))
        # logging.debug("bootstrap error: {} reported central {}".format(np.std(bootstraps), error))
        # assert(np.isclose(np.mean(bootstraps), central))
        # assert(np.isclose(np.std(bootstraps), error))

        return (ensemble, heavymass, obs, central, error), bootstraps

    def get_data(self, data_type):

        if data_type == "ainv":
            return {k.split('_')[0]:v for (k,v) in self.data.iteritems()}

        if data_type == "fDsqrtmD":

            return {k.replace("_fD",""): v*np.sqrt(self.data[k.replace("fD","mD")]) for (k,v) in self.data.iteritems() if "_fD_" in k}

        if data_type == "1/metac":
            return {k.replace("_metac",""):1.0/v for (k,v) in self.data.iteritems() if "metac" in k}

        if data_type == "1/mD":
            return {k.replace("_mD",""):1.0/v for (k,v) in self.data.iteritems() if "_mD_" in k}

        return {k.replace("_"+data_type,""):v for (k,v) in self.data.iteritems() if data_type in k}



def uk_data_tests():

    reader = uk_data_reader("/home/bfahy/data_Z2noise_runs/uk_data/UKQCD")

    # print reader.get_data("mpi")
    # print reader.get_data("fD")
    #print reader.get_data("ainv")

    import matplotlib.pyplot as plt
    import plot_uk_data
    fig, axe = plt.subplots(1)
    legend_handles = plot_uk_data.add_uk_plot_data(axe, "mpi", "mpi")
    axe.legend(handles=legend_handles, loc=0, fontsize=40, numpoints=1)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute alpha_s at 1-4 loops")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    uk_data_tests()

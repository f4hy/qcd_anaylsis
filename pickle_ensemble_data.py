#!/usr/bin/env python2
import logging
import argparse
import pandas as pd
import re
import numpy as np
from data_params import  data_params, ensemble_params, bootstrap_data

from residualmasses import residual_mass
import glob
from alpha_s import get_Cmu_mbar
import cPickle as pickle


def read_pickle(ensemble, fittype="uncorrelated"):

    filename = "{}/{}.pickle".format(ensemble,fittype)
    logging.info("reading {}".format(filename))

    return pickle.load( open( filename, "rb" ) )

def pickle_ensemble(ensemble, fittype="uncorrelated"):

    fitdatafiles = glob.glob(ensemble + "/*fit_{}_*/*.boot".format(fittype))

    data = {}
    for i in fitdatafiles:
        bsd = bootstrap_data(i)
        key = repr(bsd.dp)
        if key in data:
            logging.error("Found duplicate data key")
            exit(-1)
        data[key] = bsd

    logging.info("From {} files created {} bootstrap_data obejcts".format(len(data), len(fitdatafiles)))

    if len(data) != len(fitdatafiles):
        logging.error("not all files made it to data dictionary!!")
        exit(-1)

    output_filename = "{}/{}.pickle".format(ensemble,fittype)
    with open(output_filename, 'wb') as pdata:
        logging.info("Pickling ensembledata to {}".format(output_filename))
        pickle.dump(data, pdata, protocol=pickle.HIGHEST_PROTOCOL)


def rebuild_pickled_db(fittype="uncorrelated"):

    dirs = glob.glob("SymDW_sHtTanh_b2.0_smr3_*")
    for d in dirs:
        pickle_ensemble(d, fittype=fittype)

def test():
    #fitdatafiles = glob.glob("SymDW_sHtTanh_b2.0_smr3_*/*fit_uncorrelated_*/*.boot")
    dirs = glob.glob("SymDW_sHtTanh_b2.0_smr3_*")
    print dirs
    for d in dirs:
        data = read_pickle(d)
        print data
        for k in data:
            if "None" in k:
                print "None in key"
                exit(-1)
            if "__" in k:
                print "double _ in key"
                exit(-1)
        #print data.keys()
        for k in data.keys()[0:5]:
            print d, data[k].filename

if __name__ == "__main__":

    fittypes = ["uncorrelated", "fullcorrelated", "singlecorrelated", "interpolated"]

    parser = argparse.ArgumentParser(description="pickle all the fit data")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--fittype", required=False, choices=fittypes,
                        help="which fittype to use", default="uncorrelated")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


    rebuild_pickled_db(args.fittype)
    #test()

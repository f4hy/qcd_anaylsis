#!/usr/bin/env python2
import logging
import argparse
import pandas as pd
import re
import numpy as np
import os
import copy

from data_params import  data_params, ensemble_params, bootstrap_data

from residualmasses import residual_mass
import glob
from alpha_s import get_Cmu_mbar
import cPickle as pickle


basedir = "interpolated_to_strange_fine"

def renorm_ms_mass(ep):
    return ep.scale*((ep.s_mass + ep.residual_mass ) / ep.Zs)



def interpolate_ensemble_pair(ensemble_base, pairs, target, fittype="uncorrelated"):

    print ensemble_base
    print len(pairs)
    if len(pairs) != 2:
        logging.error("{} does not have strange pairs".format(ensemble_base))
        return

    print pairs

    e1, e2 = pairs
    filename1 = "{}/{}.pickle".format(e1,fittype)
    filename2 = "{}/{}.pickle".format(e2,fittype)
    logging.info("reading {}".format(filename1))
    ed1 = pickle.load( open( filename1, "rb" ) )
    ep1 = ensemble_params(e1)

    logging.info("reading {}".format(filename2))
    ed2 = pickle.load( open( filename2, "rb" ) )
    ep2 = ensemble_params(e2)

    print ep1
    print ep2
    ms1 = renorm_ms_mass(ep1)
    ms2 = renorm_ms_mass(ep2)

    x = (target - ms2) / (ms1 - ms2)
    print target
    new_bare_smass = ep1.s_mass*x + ep2.s_mass*(1-x)

    keys2 = [k.replace(str(ep1.s_mass), str(ep2.s_mass)).replace(ep1.ename, ep2.ename) for k in ed1.keys()]

    for k in keys2:
        if k not in ed2.keys():
            print "key2 not found", k

    new_data = {}
    for k in ed1.keys():
        k2 = k.replace(str(ep1.s_mass), str(ep2.s_mass)).replace(ep1.ename, ep2.ename)
        interpolated = (ed1[k].values)*x + (ed2[k2].values)*(1-x)
        newkey = k.replace(str(ep1.s_mass), str(new_bare_smass)).replace(ep1.ename, ep1.ename+"s")
        new_dp = copy.copy(ed1[k].dp)
        new_dp.s_mass = new_bare_smass
        new_data[newkey] = bootstrap_data(new_dp, interpolated)

    new_ensemble = ensemble_base + "_ms{0:.4f}".format(new_bare_smass)
    outdir = "{}/{}".format(basedir, new_ensemble)
    output_filename = "{}/{}.pickle".format(outdir, fittype)
    if not os.path.exists(outdir):
        logging.info("directory for output {} does not exist,"
                     "atempting to create".format(outdir))
        if outdir is not "":
            os.makedirs(outdir)

    with open(output_filename, 'wb') as pdata:
        logging.info("Pickling ensembledata to {}".format(output_filename))
        pickle.dump(new_data, pdata, protocol=pickle.HIGHEST_PROTOCOL)


def interpolate_pickle(fittype="uncorrelated"):



    dirs = glob.glob("SymDW_sHtTanh_b2.0_smr3_*")
    dirs_without_strange =  ["_".join(d.split("_")[:-1]) for d in dirs]

    target = renorm_ms_mass(ensemble_params("SymDW_sHtTanh_b2.0_smr3_64x128x08_b4.47_M1.00_mud0.0030_ms0.0150"))

    print dirs_without_strange
    print set(dirs_without_strange)
    for d in dirs_without_strange:
        interpolate_ensemble_pair(d, [p for p in dirs if p.startswith(d) ], target, fittype)

def test():
    #fitdatafiles = glob.glob("SymDW_sHtTanh_b2.0_smr3_*/*fit_uncorrelated_*/*.boot")
    dirs = glob.glob("SymDW_sHtTanh_b2.0_smr3_*")
    for d in dirs:
        data = read_pickle(d)
        for k in data.keys():
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

    fittypes = ["uncorrelated", "fullcorrelated", "singlecorrelated"]

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


    interpolate_pickle(args.fittype)
    #test()

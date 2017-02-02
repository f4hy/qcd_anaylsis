#!/usr/bin/env python2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import logging
import argparse
import os
import glob
import pandas as pd

import numpy as np

from residualmasses import residual_mass, residual_mass_errors

from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from physical_values import phys_eta, phys_etac, phys_FK, phys_mhq
from data_params import Zs, Zv

from ensemble_data1_0.ensemble_data import ensemble_data, NoStrangeInterp

from commonplotlib.auto_key import auto_key

from add_chiral_fits import add_chiral_fit

from itertools import cycle

def write_header(outwrite, dp, obsname, cval, cerr):

    outwrite.write("######### Ensemble{}\n".format(dp.ename))
    outwrite.write("######### Observable: {}\n".format(obsname))
    outwrite.write("######### central values:\n")
    outwrite.write("{:.12f}\n".format(cval))
    outwrite.write("######### error values:\n")
    outwrite.write("{:.12f}\n".format(cerr))
    outwrite.write("######### Bootstrap Samples:\n")

def myformat(i):
    return "{:.12f}".format(i)


def format_for_uk(fname, function, fpath, options):

    files = glob.glob("Sy*/{}".format(fpath))

    for f in files:
        dp = data_params(f)
        ed = ensemble_data(dp)
        data = getattr(ed, function)()
        outpath = options.output_stub + "/{0}/{0}_{1}".format(dp.ename, fname)

        if dp.heavyness != "ll":
            outpath = options.output_stub + "/{0}/{0}_{1}_mh_{2}".format(dp.ename, fname, dp.heavyq_mass)


        outdir = os.path.dirname(outpath)
        if not os.path.exists(outdir):
            logging.info("directory for output {} does not exist,"
                         "atempting to create".format(outdir))
            if outdir is not "":
                os.makedirs(outdir)
        #print data
        print data.mean()
        print data.std()
        logging.info("writeing to {}".format(outpath))

        with open(outpath, 'w') as outfile:
            write_header(outfile, dp, fname, data.mean(), data.std())
            outfile.write(",".join(map(myformat, data.values)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="average data files")

    axis_choices = ["mud", "mud_s", "mpi", "mpisqr", "2mksqr-mpisqr", "mpisqr/mq", "xi", "mq"]
    legend_choices = ["betaLs", "betaL", "heavy", "smearing", "flavor", "strange", "betaheavy"]

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")



    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)



    if args.output_stub:
        outdir = os.path.dirname(args.output_stub)
        if not os.path.exists(outdir):
            logging.info("directory for output {} does not exist,"
                         "atempting to create".format(outdir))
            if outdir is not "":
                os.makedirs(outdir)

    obs = {}
    obs["mpi"] = ("pion_mass" , "simul_fixed_fit_uncorrelated_ud-ud/simul_fit_uncorrelated_ll_ud-ud_0_1-1_1_PP.boot")
    obs["mK"] = ("kaon_mass" , "simul_fixed_fit_uncorrelated_ud-ud/simul_fit_uncorrelated_ll_ud-ud_0_1-1_1_PP.boot")

    obs["mD0"] = ("D_mass", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m0_heavy-ud_0_1-1_1_PP.boot")
    obs["mDs0"] = ("Ds_mass", "simul_fixed_fit_uncorrelated_heavy-s/simul_fit_uncorrelated_m0_heavy-s_0_1-1_1_PP.boot")
    obs["fD0"] = ("fD", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m0_heavy-ud_0_1-1_1_PP.boot")
    obs["fDs0"] = ("fDs", "simul_fixed_fit_uncorrelated_heavy-s/simul_fit_uncorrelated_m0_heavy-s_0_1-1_1_PP.boot")

    obs["mD1"] = ("D_mass", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m1_heavy-ud_0_1-1_1_PP.boot")
    obs["mDs1"] = ("Ds_mass", "simul_fixed_fit_uncorrelated_heavy-s/simul_fit_uncorrelated_m1_heavy-s_0_1-1_1_PP.boot")
    obs["fD1"] = ("fD", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m1_heavy-ud_0_1-1_1_PP.boot")
    obs["fDs1"] = ("fDs", "simul_fixed_fit_uncorrelated_heavy-s/simul_fit_uncorrelated_m1_heavy-s_0_1-1_1_PP.boot")

    obs["mD2"] = ("D_mass", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m2_heavy-ud_0_1-1_1_PP.boot")
    obs["mDs2"] = ("Ds_mass", "simul_fixed_fit_uncorrelated_heavy-s/simul_fit_uncorrelated_m2_heavy-s_0_1-1_1_PP.boot")
    obs["fD2"] = ("fD", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m2_heavy-ud_0_1-1_1_PP.boot")
    obs["fDs2"] = ("fDs", "simul_fixed_fit_uncorrelated_heavy-s/simul_fit_uncorrelated_m2_heavy-s_0_1-1_1_PP.boot")

    obs["metac0"] = ("HH_mass", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m0_heavy-ud_0_1-1_1_PP.boot")
    obs["metac1"] = ("HH_mass", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m1_heavy-ud_0_1-1_1_PP.boot")
    obs["metac2"] = ("HH_mass", "simul_fixed_fit_uncorrelated_heavy-ud/simul_fit_uncorrelated_m2_heavy-ud_0_1-1_1_PP.boot")

    for fname, d in obs.iteritems():
        fun, files = d
        format_for_uk(fname, fun, files, args)

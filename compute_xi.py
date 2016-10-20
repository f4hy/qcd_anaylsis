#!/usr/bin/env python2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import logging
import argparse
from matplotlib.widgets import CheckButtons
import os
import pandas as pd
import math

from cStringIO import StringIO
import numpy as np
import re

from residualmasses import residual_mass

import plot_helpers

from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq
from data_params import Zs, Zv

from auto_key import auto_key


def compute_xi(options):
    """
    Compute xi which is  = M_pi^2 / (8 pi^2 f_pi^2)
    """

    fit_file_cols = ["bootstrap", "mass", "amp1", "amp2"]
    mpi_df = pd.read_csv(options.fitmassfile, comment='#', names=fit_file_cols)
    mpi_data = mpi_df["mass"]

    decay_file_cols = ["decay"]
    decay_df = pd.read_csv(options.decayconstantfile, comment='#', names=decay_file_cols)
    decay_data = decay_df["decay"]

    mpisqr_data = mpi_data**2
    decaysqr_data = decay_data**2

    xi = mpisqr_data / (decaysqr_data * 8 * np.pi**2)

    count = 0
    if options.output_stub:
        outfilename = options.output_stub + ".out"
        logging.info("writing to {}".format(outfilename))
        with open(outfilename, 'w') as outfile:
            outfile.write("#xi, {}, {}\n".format(xi.mean(), xi.std()))
            for d in xi:
                count += 1
                outfile.write("{}\n".format(d))
    else:
        print "Xi mean: ", xi.mean()
        print "Xi std: ", xi.std()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="average data files")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-F", "--decayconstantfile", type=argparse.FileType('r'), required=False,
                        help="Decay constant file")
    parser.add_argument("-M", "--fitmassfile", type=argparse.FileType('r'), required=False,
                        help="Fitted mass file")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


    compute_xi(args)

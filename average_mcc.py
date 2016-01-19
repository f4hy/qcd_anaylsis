#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import glob


def average_mcc(options):

    files = glob.glob(options.directory+"/S*{}*.mcc".format(options.beta))
    if len(files ) < 1:
        logging.error("No mcc files found")
        exit(-1)

    filehandles = [open(f) for f in files]
    txts = [f.read() for f in filehandles]
    values = [float(f.split("/")[0].strip(" +-")) for f in txts]
    mean = np.mean(values)
    # mean_err = np.mean(errors)
    # print mean
    with open("{}/{}.mcc".format(options.directory, options.beta), "w") as outfile:
        outfile.write("{}\n".format(mean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to interpolate the heavy mass")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="directory with mcc files")
    parser.add_argument("-b", "--beta", type=str, required=True,
                        help="beta")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    average_mcc(args)

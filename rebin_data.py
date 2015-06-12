#!/usr/bin/env python2
import logging
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import leastsq
import glob
import math

def rebin_data(options):
    """ rebin jackknifed data """

    for f in options.files:
        print f
        df = pd.read_csv(f, comment='#', names=["flow"])
        print df
        print df["flow"].mean()

        N = len(df)
        print N

        s = math.fsum(df["flow"])

        print s
        originals =  (-1)*(df["flow"]*(N-1)-s)
        print originals
        print len(originals[0:14])
        print originals[0:14].mean()
        print (s - originals[0:14].mean())/(N-1)
        #exit(-1)

        binsize = N/options.bins

        binned = []
        for i in range(options.bins):
            start, end = i*binsize,(i+1)*binsize
            new = originals[start:end].mean()
            binned.append(new)


        print binned

        ns = math.fsum(binned)
        outfile = options.output + "/" + f.replace("jack_bin1", "binned_to_{}".format(options.bins))
        logging.info("writing rebinned data to {}".format(outfile))

        with open(outfile, "w") as ofile:
            for b in binned:
                ofile.write("{}\n".format((ns - b)/(len(binned)-1)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script to rebin jackknifed data")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-b", "--bins", type=int, required=True,
                        help="convert to N bins")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    rebin_data(args)

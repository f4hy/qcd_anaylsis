#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import glob

heavymap = {}
heavymap[("4.17", "m0")] = 0.35
heavymap[("4.17", "m1")] = 0.445
heavymap[("4.17", "m2")] = 0.52
heavymap[("4.35", "m0")] = 0.20
heavymap[("4.35", "m1")] = 0.273
heavymap[("4.35", "m2")] = 0.31
heavymap[("4.47", "m0")] = 0.150
heavymap[("4.47", "m1")] = 0.206
heavymap[("4.47", "m2")] = 0.250

scale = {"4.17": 2490, "4.35": 3660, "4.47": 4600}


def read_files(basefile, spinaverage=None):
    data = {}

    col_names = ["config", "mass", "amp1", "amp2"]
    if "decay_const" in basefile:
        col_names = ["mass"]


    if "m0" not in basefile:
        raise SystemExit("interpolate requires inputing the m0 mass file")

    m1file = re.sub("_heavy0\.[0-9]+_", "_*", basefile.replace("_m0_", "_m1_"))
    m1file = glob.glob(m1file)[0]
    m2file = re.sub("_heavy0\.[0-9]+_", "_*", basefile.replace("_m0_", "_m2_"))

    m2file = glob.glob(m2file)[0]

    files = [basefile, m1file, m2file]

    for f in files:
        logging.info("reading {}".format(f))

        beta = re.search("_b(4\.[0-9]*)_", f).group(1)
        heavyness = re.search("_([a-z][a-z0-9])_", f).group(1)

        df = pd.read_csv(f, comment='#', names=col_names)

        if spinaverage:
            if "vectorave" not in f:
                raise SystemExit("spin average requires inputing the vector file")
            ppfile = f.replace("vectorave", "PP")
            PPdf = pd.read_csv(ppfile, comment='#', names=col_names)
            data[heavymap[(beta, heavyness)]] = (3.0*df["mass"] + PPdf["mass"]) / 4.0

        else:
            data[heavymap[(beta, heavyness)]] = df["mass"]
    return data


def get_physical_point(physical, line_params):
    m, b = line_params
    mcc = (physical - b)/m
    return mcc


def interpolate(data, physical=None):

    cmass = []
    mmasses = []

    for heavymass, mesonmass in data.iteritems():
        cmass.append(heavymass)
        mmasses.append(mesonmass)

    def line(v, x, y):
        return (v[0]*x+v[1]) - y

    Nconfigs = len(mmasses[0])

    A = np.array(cmass)
    fits = []
    for i in range(Nconfigs):
        mmass = [m[i] for m in mmasses]
        B = np.array(mmass)


        slope_guess = (max(mmass)-min(mmass)) / (max(cmass) - min(cmass))
        int_guess = min(mmass) - min(cmass)*slope_guess
        guess = [slope_guess, int_guess]
        logging.info("guessing a line with y={}x+{}".format(*guess))
        best_fit, _, info, mesg, ierr = leastsq(line, guess, args=(A, B), maxfev=10000, full_output=True)
        fits.append(best_fit)

    return fits


def gen_line_func(parameters):
    def fitline(x):
        return parameters[0] * x + parameters[1]

    return fitline


def plot_fitline(data, fitline, m_cc, physical, outstub):
    size = 100

    for heavymass, mesonmass in data.iteritems():
        plt.scatter(heavymass, mesonmass.median(), s=size)
        #plt.scatter(heavymass, mesonmass, s=size)

    xdata = np.arange(min(data.keys())-0.005, max(data.keys())+0.005, 0.001)
    ydata = [fitline(x) for x in xdata]
    plt.plot(xdata, ydata)

    if physical:
        logging.info("physical point {}, {}".format(m_cc, physical))

        plt.scatter(m_cc, physical, c='r', s=size)

    if outstub is not None:
        filename = outstub+".png"
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


def write_data(m_cc, output_stub, suffix):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing mcc to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        ofile.write("{}\n".format(m_cc))

def write_shifted_data(shifted_data, mcc, output_stub, suffix):
    if output_stub is None:
        logging.info("Not writing output")
        logging.info("shifted data median:{} and std:{}".format(shifted_data.mean(), shifted_data.std()))
        return
    outfilename = output_stub + suffix
    logging.info("writing shifted data to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        ofile.write("#{}, {}, {}\n".format(mcc, shifted_data.mean(), shifted_data.std()))
        for i,d in enumerate(shifted_data):
            ofile.write("{}, {}\n".format(i, d))
    return


def interpolate_heavymass(options):
    """ script to interpolate the heavy mass """
    logging.debug("Called with {}".format(options))

    beta = re.search("_b(4\.[0-9]*)_", options.basefile).group(1)

    data = read_files(options.basefile, spinaverage=options.spinaverage)


    line_params = interpolate(data)
    line_params = np.median(line_params, axis=0)

    fitline = gen_line_func(line_params)

    logging.info("Using physical point {} to set the scale".format(options.physical))

    m_cc = get_physical_point(options.physical/scale[beta], line_params)

    plot_fitline(data, fitline, m_cc, options.physical/scale[beta], options.output_stub)

    write_data(m_cc, options.output_stub, ".mcc")



def shift_data(options):
    """ script to interpolate the heavy mass """
    logging.debug("Called with {}".format(options))

    beta = re.search("_b(4\.[0-9]*)_", options.basefile).group(1)

    data = read_files(options.basefile, spinaverage=options.spinaverage)

    for k,v in data.iteritems():
        logging.debug("original data: {} has median: {}".format(k, v.median()))

    line_params = interpolate(data)

    scaled_values = []
    count = 0

    shifted_data = np.array([gen_line_func(lp)(options.mcc) for lp in line_params])
    write_shifted_data(shifted_data, options.mcc, options.output_stub, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to interpolate the heavy mass")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("--spinaverage", action="store_true",
                        help="spinaverage vector with pseudoscalar")
    parser.add_argument("-p", "--physical", type=float,
                        help="add physical point")
    parser.add_argument("-m", "--mcc", type=float,
                        help="scale data using given heavy mass")
    parser.add_argument('basefile', metavar='f', type=str,
                        help='input file')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.physical:
        interpolate_heavymass(args)
    elif args.mcc:
        shift_data(args)
    else:
        logging.error("Either mcc or physical point must be given")

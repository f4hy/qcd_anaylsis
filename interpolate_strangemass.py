#!/usr/bin/env python2
import logging
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import plot_helpers

from data_params import scale, data_params, read_fit_mass, phys_pion, phys_kaon

colors = ['b', 'r', 'k', 'm', 'c', 'y', 'b', 'r', 'k', 'm', 'c', 'y']


def read_files(files, fitdata):
    data = {}

    col_names = ["bootstrap", "mass"]

    s_masses = set(float(re.search("ms([0-9]\.[0-9]*)", f).group(1)) for f in files)

    logging.info("Found two strange masses {}".format(", ".join(map(str, s_masses))))

    data = {s: {} for s in s_masses}

    for s in s_masses:
        pass

    for f in files:
        logging.info("reading {}".format(f))
        dp = data_params(f)

        s_mass = float(re.search("ms([0-9]\.[0-9]*)", f).group(1))

        df = pd.read_csv(f, comment='#', names=col_names, index_col=0)

        pion_masses = read_fit_mass(dp, "ud-ud", fitdata)

        pion_mass_sqr = pion_masses.mean()**2
        data[s_mass][pion_mass_sqr] = df

    return data


def get_physical_point(physical, line_params):
    m, b = line_params
    mcc = (physical - b)/m
    return mcc


def interpolate(data, physical=None, debug=False):

    pisqr_masses = []
    mmasses = []
    stds = []


    for pisqr_mass, mesonmass in data.iteritems():
        pisqr_masses.append(pisqr_mass)
        mmasses.append(np.mean(mesonmass.values))
        stds.append(np.std(mesonmass))

    def line(v, x, y):
        return (v[0]*x+v[1]) - y


    A = np.array(pisqr_masses)
    fits = []

    #for i in range(Nconfigs):
    #mmass = [m["mass"][i] for m in mmasses]
    B = np.array(mmasses)


    slope_guess = (max(mmasses)-min(mmasses)) / (max(pisqr_masses) - min(pisqr_masses))
    int_guess = min(mmasses) - min(pisqr_masses)*slope_guess
    guess = [slope_guess, int_guess]
    logging.info("guessing a line with y={}x+{}".format(*guess))
    best_fit, _, info, mesg, ierr = leastsq(line, guess, args=(A, B),
                                            maxfev=10000, full_output=True)
    logging.info("found a line with y={}x+{}".format(*best_fit))
    fits.append(best_fit)

    if debug:
        plt.scatter(A, B)
        xdata = np.arange(0, 0.05, 0.001)
        ydata = [best_fit[0]*x+best_fit[1] for x in xdata]
        plt.plot()
        plt.plot(xdata, ydata)
        plt.show()


    return fits


def gen_line_func(parameters):
    def fitline(x):
        return parameters[0] * x + parameters[1]

    return fitline


def plot_fitline(data, fitline, label):

    miny = float("inf")
    maxy = float("-inf")
    color = colors.pop()

    for heavymass, mesonmass in data.iteritems():
        errs = plot_helpers.error(mesonmass.values)
        plt.errorbar(heavymass, mesonmass.mean(), yerr=errs, ms=8, c=color)
        miny = min(miny, mesonmass.mean().values)
        maxy = max(maxy, mesonmass.mean().values)

    xdata = np.arange(0.0, max(data.keys())+0.005, 0.001)
    ydata = [fitline(x) for x in xdata]
    plt.plot(xdata, ydata, c=color, label=label)


def display_plots(outstub, physical_x, physical_y, beta):
    size = 100
    color = colors.pop()
    if physical_y:
        logging.info("physical point {}, {}".format(physical_x, physical_y))

        plt.scatter(physical_x, physical_y, c=color, s=size)

    plt.plot([physical_x]*2, [0, 100], label="physical", c=color, scaley=False)

    plt.legend(loc=0)

    plt.xlabel("$m_\pi^2$")
    plt.ylabel("$2m_k^2-m_\pi^2 \propto m_s$")
    plt.title(r"Strange quark mass interpolation $\beta = {}$".format(beta))

    if outstub is not None:
        filename = outstub+".png"
        if args.eps:
            filename = outstub+".pdf"
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


def write_weights(weights, output_stub, suffix):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing strangeweights to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        ofile.write("{}, {}\n".format(*weights))


def write_smass(smass, output_stub, suffix):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing smass to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        ofile.write("{}\n".format(smass))


def interpolate_strangemass(options):
    """ script to interpolate the heavy mass """
    logging.debug("Called with {}".format(options))

    beta = re.search("_b(4\.[0-9]*)_", options.files[0]).group(1)
    if not all([beta in f for f in options.files]):
        raise RuntimeError("Not all the same beta")

    physical_pisqr = phys_pion**2/(scale[beta]**2)
    physical_s = (2*(phys_kaon**2)-phys_pion**2)/(scale[beta]**2)

    alldata = read_files(options.files, options.fitdata)


    fitted_s = {}
    for strangeness, data in alldata.iteritems():
        line_params = interpolate(data)
        line_params = np.mean(line_params, axis=0)

        fitline = gen_line_func(line_params)

        fit_s = fitline(physical_pisqr)

        fitted_s[strangeness] = fit_s

        plot_fitline(data, fitline, "$m_s={}$".format(strangeness))

    y1, y2 = (fitted_s[i] for i in sorted(fitted_s))
    weight = (physical_s - y2) / (y1-y2)
    weights = (weight, 1-weight)

    display_plots(options.output_stub, physical_pisqr, physical_s, beta)

    write_weights(weights, options.output_stub, ".strangeweights")

    s1,s2 = alldata.keys()
    if s1 > s2:
        smass = weights[1]*s1 + weights[0]*s2
    else:
        smass = weights[0]*s1 + weights[1]*s2
    write_smass(smass, options.output_stub, ".smass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to interpolate the heavy mass")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-ms", type=float,
                        help="scale data using given strange mass")
    parser.add_argument("-e", "--eps", action="store_true",
                        help="save as eps not png")
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    interpolate_strangemass(args)

#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from iminuit import Minuit

from residualmasses import residual_mass, residual_mass_errors

from data_params import data_params, read_fit_mass, scale, phys_pion, phys_kaon, phys_Fpi
from data_params import Zs, Zv
from physical_values import phys_pionplus

from ensemble_data1_0.ensemble_data import ensemble_data, MissingData

import inspect
import collections

from plotter1_0.get_data import get_data


def interpolate_from_global_fit(fitfile, Lambda, outstub):

    vals = {}
    errs = {}
    with open(fitfile) as f:
        for line in f:
            print line
            if line.startswith("#"):
                continue
            name, values = line.split(",")
            val = float(values.split()[0])
            err = float(values.split()[-1])
            vals[name] = val
            errs[name] = err

    print vals


    if "fdssqrtms_chiral_dmss_HQET" in fitfile:
        data = fdssqrtms_chiral_dmss_HQET(vals, Lambda, outstub)

    if outstub:
        with open(outstub + ".out", "w") as outfile:
            for x,y in data.iteritems():
                outfile.write("{}, {}\n".format(x, y))

    color = 'k'
    plotsettings = dict(linestyle="none", c=color, marker='o',
                        ms=15, elinewidth=4,
                        capsize=8, capthick=2, mec=color, mew=2,
                        aa=True, fmt='o', ecolor=color)


    for x,y in data.iteritems():
        plt.errorbar(x,y,yerr=0, **plotsettings)

    plt.ylim(0.7, 1.4)
    plt.show()

def fdssqrtms_chiral_dmss_HQET(values, Lambda, outstub):

    values["C1"] *= 1000.0
    values["C2"] *= (1000.0)**2
    values["gamma"] *= 1.0/(10000.0)
    values["eta"] *= 1.0/(100.0)
    values["mu"] *= 0.001


    Fssqrtms_inf = values["Fssqrtms_inf"]
    C1 = values["C1"]
    C2 = values["C2"]
    gamma = values["gamma"]
    eta = values["eta"]
    mu = values["mu"]

    m = 1968.3
    m = m/2
    prevf = 0

    data = {}

    # x = np.linspace(0, 0.0007)
    # print x
    # print Fssqrtms_inf*(1 + C1 * x + C2 * x**2 )
    # plt.errorbar(x, Fssqrtms_inf*(1 + C1 * x + C2 * x**2 ), yerr=0)
    # plt.ylim(5000,22000)
    # plt.show()
    # exit(-1)

    for i in range(40):
        #print m
        newm = Lambda*m
        mgev = newm/1000
        #print 1/mgev
        mDs_inv = 1/newm
        f = Fssqrtms_inf*(1 + C1 * mDs_inv + C2 * mDs_inv**2 )
        if prevf > 0:
            print "L ratio", np.sqrt(Lambda) * f/prevf
            print "ratio", f/prevf
            print "ratio by L", (f/prevf) / np.sqrt(Lambda)
            print prevm
            data[1.0/prevm] = f/prevf
        prevf = f
        prevm = m
        m = newm
        #print "f", f
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="interpolate global fit file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-l", "--Lambda", type=float, required=True,
                        help="ratio to use")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('fitfile', metavar='f', type=str,
                        help='file with fit parameters')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    interpolate_from_global_fit(args.fitfile, args.Lambda, args.output_stub)

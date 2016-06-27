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

from plot_helpers import print_paren_error

from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from ensamble_info import all_same_beta, all_same_heavy, all_same_flavor
from ensamble_info import phys_pion, phys_kaon, phys_mq, phys_Fpi
from ensamble_info import Zs, Zv

from auto_key import auto_key


def convert_constants(chiral_fit_file, options):

    values = {}
    errors = {}

    if "failed" in chiral_fit_file.name:
        return

    print chiral_fit_file.name
    for i in chiral_fit_file:
        if i.startswith("#"):
            fittype = i.split(" ")[0][1:]
            continue

        name, val, err = (j.strip() for j in i.replace("+/-",",").split(","))
        values[name] = float(val)
        errors[name] = float(err)

    try:
        values[" M_\pi<"] = re.search("cut([0-9]+)", chiral_fit_file.name).group(1)
        errors[" M_\pi<"] = 0
    except:
        pass


    print values
    # if "c3" in values.keys():

    #     print values["c3"] / phys_Fpi
    #     print (8*np.pi**2 * phys_Fpi**2) / (np.exp(values["c3"] / phys_Fpi))
    #     print "LAMBDA3",  np.sqrt((8*np.pi**2 * phys_Fpi**2) / (np.exp(values["c3"] / phys_Fpi)))

    if "c3" in values.keys():

        B = values["B"]

        LAMBDA3 = np.sqrt((8*np.pi**2 * phys_Fpi**2) / (np.exp(values["c3"] / B)))
        LAMBDA3_err = errors["c3"] * (LAMBDA3 / (2*phys_Fpi) )

        c3string = print_paren_error(values["c3"], errors["c3"])
        c3percent = 100*errors["c3"]/ values["c3"]
        lam3string = print_paren_error(LAMBDA3, LAMBDA3_err)
        lam3percent = 100*LAMBDA3_err/LAMBDA3

        l3 = np.log(LAMBDA3**2 / phys_pion**2)
        l3_err = LAMBDA3_err *2 / LAMBDA3
        l3string = print_paren_error(l3, l3_err)
        l3percent = 100*l3_err/l3

        logging.info("c3: {}, {}%".format(c3string, c3percent ))
        logging.info("lambda3: {}, {}%".format(lam3string, lam3percent ))
        logging.info("l3: {}, {}%\n".format(l3string, l3percent ))


    if "c4" in values.keys():

        f = values["F_0"]

        LAMBDA4 = np.sqrt((8*np.pi**2 * phys_Fpi**2) / (np.exp(values["c4"] / f)))
        LAMBDA4_err = errors["c4"] * (LAMBDA4 / (2*phys_Fpi) )

        c4string = print_paren_error(values["c4"], errors["c4"])
        c4percent = 100*errors["c4"]/ values["c4"]
        lam4string = print_paren_error(LAMBDA4, LAMBDA4_err)
        lam4percent = 100*LAMBDA4_err/LAMBDA4

        l4 = np.log(LAMBDA4**2 / phys_pion**2)
        l4_err = LAMBDA4_err *2 / LAMBDA4
        l4string = print_paren_error(l4, l4_err)
        l4percent = 100*l4_err/l4

        logging.info("c4: {}, {}%".format(c4string, c4percent ))
        logging.info("lambda4: {}, {}%".format(lam4string, lam4percent ))
        logging.info("l4: {}, {}%\n".format(l4string, l4percent ))


    if "Lambda4" in values.keys():

        LAMBDA4, LAMBDA4_err = values["Lambda4"], errors["Lambda4"]
        c4 = phys_Fpi * np.log((8*np.pi**2 * phys_Fpi**2)/(values["Lambda4"]**2))
        c4_err = errors["Lambda4"] * np.abs((2*phys_Fpi) / values["Lambda4"])
        lam4string = print_paren_error(values["Lambda4"], errors["Lambda4"])
        lam4percent = 100*errors["Lambda4"] / values["Lambda4"]
        c4string = print_paren_error(c4, c4_err)
        c4percent = 100*c4_err / np.abs(c4)


        l4 = np.log(LAMBDA4**2 / phys_pion**2)
        l4_err = LAMBDA4_err *2 / LAMBDA4
        l4string = print_paren_error(l4, l4_err)
        l4percent = 100*l4_err/l4

        logging.info("lambda4: {}, {}%".format(lam4string, lam4percent ))
        logging.info("c4: {}, {}%".format(c4string, c4percent ))
        logging.info("l4: {}, {}%\n".format(l4string, l4percent ))

    if "Lambda3" in values.keys():

        B = values["B"]

        LAMBDA3, LAMBDA3_err = values["Lambda3"], errors["Lambda3"]
        c3 = B * np.log((8*np.pi**2 * phys_Fpi**2)/(values["Lambda3"]**2))
        c3_err = errors["Lambda3"] * np.abs((2*B) / values["Lambda3"])
        lam3string = print_paren_error(values["Lambda3"], errors["Lambda3"])
        lam3percent = 100*errors["Lambda3"] / values["Lambda3"]
        c3string = print_paren_error(c3, c3_err)
        c3percent = 100*c3_err / np.abs(c3)


        l3 = np.log(LAMBDA3**2 / phys_pion**2)
        l3_err = LAMBDA3_err *2 / LAMBDA3
        l3string = print_paren_error(l3, l3_err)
        l3percent = 100*l3_err/l3

        logging.info("lambda3: {}, {}%".format(lam3string, lam3percent ))
        logging.info("c3: {}, {}%".format(c3string, c3percent ))
        logging.info("l3: {}, {}%\n".format(l3string, l3percent ))

    if "B" in values.keys():

        B, B_err = values["B"], errors["B"]
        SIGMA = (B*phys_Fpi**2)/2.0
        SIGMA_err = (B_err*phys_Fpi**2)/2.0
        Sroot = SIGMA**(1.0/3.0)
        Sroot_err = B_err *((phys_Fpi**2)/2.0) / (3.0* SIGMA**(2.0/3.0))

        Bstring = print_paren_error(B, B_err)
        Bpercent = 100*B_err/B
        SIGMAstring = print_paren_error(SIGMA, SIGMA_err)
        SIGMApercent = 100*SIGMA_err/SIGMA
        Srootstring = print_paren_error(Sroot, Sroot_err)
        Srootpercent = 100*Sroot_err/Sroot

        logging.info("B: {}, {}%".format(Bstring, Bpercent ))
        logging.info("Sroot: {}, {}%".format(Srootstring, Srootpercent ))
        logging.info("SIGMA: {}, {}%\n".format(SIGMAstring, SIGMApercent ))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="convert constants into other formats")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('files', metavar='f', type=argparse.FileType('r'), nargs='+',
                        help='file from output of chiral fit')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    for i in args.files:
        convert_constants(i, args)

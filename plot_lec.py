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

from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi
from data_params import Zs, Zv

from auto_key import auto_key

from plot_decay_constants import auto_fit_range
from add_chiral_fits import format_parameters

first = True

colors = ['b', 'r', 'k', 'm', 'c', 'y']*10

def mark_generator():
    for i in ["D", "^", "<", ">", "v", "x", "p", "8"]:
        yield i

def number_generator():
    for i in range(1,100):
        yield i

plotindex = number_generator()

summary_lines = []

def plot_constants(axe, chiral_fit_file, options):

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
    # if "c3" in values:

    #     print values["c3"] / phys_Fpi
    #     print (8*np.pi**2 * phys_Fpi**2) / (np.exp(values["c3"] / phys_Fpi))
    #     print "LAMBDA3",  np.sqrt((8*np.pi**2 * phys_Fpi**2) / (np.exp(values["c3"] / phys_Fpi)))

    if "F_0" in values:
        F_0 = values["F_0"]/np.sqrt(2)
        F_0_err = errors["F_0"]/np.sqrt(2)

    if "c3" in values:

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


    if "c4" in values:

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


    if "Lambda4" in values:

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

    if "Lambda3" in values:

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

    if "B" in values:

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


    if "x" in chiral_fit_file.name:
        if "NNLO" in chiral_fit_file.name:
            label = "NNLO x"
        elif "NLO" in chiral_fit_file.name:
            label = "NLO x"
            print chiral_fit_file.name, "NLO x"
    if "xi" in chiral_fit_file.name or "XI" in chiral_fit_file.name:
        if "NNLO" in chiral_fit_file.name:
            label = "NNLO xi"
        elif "NLO" in chiral_fit_file.name:
            label = "NLO xi"

    color = colors.pop()
    plotsettings = dict(linestyle="none", c=color, marker="o",
                        label=label, ms=12, elinewidth=4,
                        capsize=8, capthick=2, mec=color, mew=2,
                        aa=True, mfc=color, fmt='o', ecolor=color)

    try:
        if options.constant == "Lambda4":
            ydata=LAMBDA4
            y_err=LAMBDA4_err
        if options.constant == "Lambda3":
            ydata=LAMBDA3
            y_err=LAMBDA3_err
        if options.constant == "l4":
            ydata=l4
            y_err=l4_err
        if options.constant == "l3":
            ydata=l3
            y_err=l3_err
        if options.constant == "sigma13":
            ydata=Sroot
            y_err=Sroot_err
        if options.constant == "f0":
            ydata=F_0
            y_err=F_0_err
        thisx = plotindex.next()
        summary_lines.append("{}, {}, {}\n".format(label, ydata, y_err))
        return axe.errorbar(x=thisx, y=ydata, yerr=y_err, **plotsettings)
    except UnboundLocalError:
        logging.warn("file {} did not contain {}".format(chiral_fit_file.name, options.constant))
        return []

def add_others(axe, options):
    color = 'k'
    flagplotsettings = dict(linestyle="none", c=color, marker="x",
                            ms=12, elinewidth=4, label="FLAG",
                            capsize=8, capthick=2, mec=color, mew=2,
                            aa=True, mfc=color, fmt='o', ecolor=color)
    color = 'g'
    mark_gen = mark_generator()
    plotsettings = dict(linestyle="none", c=color,
                            ms=12, elinewidth=4,
                            capsize=8, capthick=2, mec=color, mew=2,
                            aa=True, mfc=color, fmt='o', ecolor=color)

    # if options.constant == "Lambda4":
    #     return axe.errorbar(x=plotindex.next(), y=LAMBDA4, yerr=LAMBDA4_err, **plotsettings)
    # if options.constant == "Lambda3":
    #     return axe.errorbar(x=plotindex.next(), y=LAMBDA3, yerr=LAMBDA3_err, **plotsettings)
    if options.constant == "l4":
        axe.set_ylim(0,6)
        axe.errorbar(x=plotindex.next(), y=4.02, yerr=0.28, **flagplotsettings)
        axe.errorbar(x=plotindex.next(), y=3.8, yerr=0.44, label="BMW13", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=3.99, yerr=0.18, label="RBC/UKQCD 12", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=4.03, yerr=0.16, label="Borsanyi 12", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=3.98, yerr=0.32, label="MILC 10A", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=4.30, yerr=0.51, label="NPLQCD 11", marker=mark_gen.next(), **plotsettings)
    if options.constant == "l3":
        axe.set_ylim(0,6)
        axe.errorbar(x=plotindex.next(), y=3.05, yerr=0.99, **flagplotsettings)
        axe.errorbar(x=plotindex.next(), y=2.5, yerr=0.64, label="BMW13", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=2.91, yerr=0.24, label="RBC/UKQCD 12", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=3.16, yerr=0.30, label="Borsanyi 12", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=2.85, yerr=0.81, label="MILC 10A", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=4.04, yerr=0.40, label="NPLQCD 11", marker=mark_gen.next(), **plotsettings)
    if options.constant == "sigma13":
        axe.set_ylim(150,350)
        axe.errorbar(x=plotindex.next(), y=271.0, yerr=15.0, **flagplotsettings)
        axe.errorbar(x=plotindex.next(), y=271.0, yerr=4.0, label="BMW13", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=281.5, yerr=5.2, label="MILC 10A", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=272.3, yerr=1.84, label="Borsanyi 12", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=234.3, yerr=17.4, label="JLQCD/TWQCD 10", marker=mark_gen.next(), **plotsettings)
    if options.constant == "f0":
        axe.set_ylim(70,100)
        # axe.errorbar(x=plotindex.next(), y=271.0, yerr=15.0, **flagplotsettings)
        axe.errorbar(x=plotindex.next(), y=88.0, yerr=1.4, label="BMW13", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=87.5, yerr=1.0, label="MILC 10A", marker=mark_gen.next(), **plotsettings)
        axe.errorbar(x=plotindex.next(), y=86.78, yerr=0.25, label="Borsanyi 12", marker=mark_gen.next(), **plotsettings)
        # axe.errorbar(x=plotindex.next(), y=234.3, yerr=17.4, label="JLQCD/TWQCD 10", marker=mark_gen.next(), **plotsettings)

    divlinex = plotindex.next()
    axe.axvline(x=divlinex, color="k")
    axe.text(divlinex+0.5, 0.5, 'This work', fontsize=80)

    return

if __name__ == "__main__":

    choices = ["Lambda4", "Lambda3", "l3", "l4", "sigma13", "f0"]

    parser = argparse.ArgumentParser(description="convert constants into other formats")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('files', metavar='f', type=argparse.FileType('r'), nargs='+',
                        help='file from output of chiral fit')
    parser.add_argument("-c", "--constant", required=False, type=str, choices=choices, default="Lambda4",
                        help="which model to use")
    parser.add_argument("-e", "--eps", action="store_true",
                        help="save as eps not png")
    parser.add_argument("--pdf", action="store_true",
                        help="save as pdf not png")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


    fig, axe = plt.subplots(1)
    axe.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    plots = []
    add_others(axe, args)
    for i in args.files:
        plots.extend(plot_constants(axe, i, args))

    fontsettings = dict(fontweight='bold', fontsize=50)

    axe.set_title("${}$".format(format_parameters(args.constant)), **fontsettings)
    axe.set_ylabel("${}$".format(format_parameters(args.constant)), **fontsettings)

    axe.tick_params(axis='y', which='major', labelsize=40)


    axe.legend(loc=0, fontsize=30, numpoints=1)
    plt.xlim(0, plotindex.next()+3)


    if(args.output_stub):
        fig.set_size_inches(26.5, 9.5)
        summaryfilename = args.output_stub + ".txt"
        logging.info("Writting summary to {}".format(summaryfilename))
        with open(summaryfilename, 'w') as summaryfile:
            for i in summary_lines:
                summaryfile.write(i)
        fig.tight_layout()
        file_extension = ".png"
        if args.eps:
            file_extension = ".eps"
        if args.pdf:
            file_extension = ".pdf"
        filename = args.output_stub + file_extension
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename)
        exit()

    plt.show()

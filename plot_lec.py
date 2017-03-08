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

from commonplotlib.plot_helpers import print_paren_error

from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon, phys_mq, phys_Fpi
from data_params import Zs, Zv

from commonplotlib.auto_key import auto_key

from plot_decay_constants import auto_fit_range
from plotter1_0.add_chiral_fits import format_parameters

from itertools import cycle

first = True

colors = cycle(reversed(['k', 'y', 'm', 'c',  'b', 'r']))

def mark_generator():
    for i in ["D", "^", "<", ">", "v", "x", "p", "8"]:
        yield i

def number_generator():
    for i in range(1,100):
        yield i*0.2

plotindex = number_generator()

summary_lines = []

def read_data(fit_file):
    infoline = fit_file.readline().strip()
    columns = infoline.split(",")[1:]
    name = infoline.strip("# ").split(",")[0]
    df = pd.read_csv(fit_file, sep=",", delimiter=",", names=columns)
    return name, df


def Lambda3(data, label=None):
    lam3 = data.Lambda3
    mean = lam3.mean()
    err =  lam3.std()
    syserr = (data.Lambda3*0.01433447098).mean()
    logging.info("{} lam3: mean{} err{} syserr{}".format(label, mean, err, syserr))
    return (mean, err, syserr)

def Lambda4(data, label=None):
    lam4 = data.Lambda4
    mean = lam4.mean()
    err =  lam4.std()
    syserr = (data.Lambda4*0.01433447098).mean()
    logging.info("{} lam4: mean{} err{} syserr{}".format(label, mean, err, syserr))
    return (mean, err, syserr)



def f0(data, label=None):
    F = data.F_0 #/ np.sqrt(2)
    mean = F.mean()
    err =  F.std()
    syserr = ((F*0.01433447098)).mean()
    logging.info("{} F: mean{} err{} syserr{}".format(label, mean, err, syserr))
    return (mean, err, syserr)

def l3(data, label=None):
    ell3 = np.log(data.Lambda3**2 / phys_pion**2)
    mean = ell3.mean()
    err =  ell3.std()
    syserr = None
    logging.info("{} l3: mean{} err{} syserr{}".format(label, mean, err, syserr))
    return (mean, err, syserr)

def l4(data, label=None):
    ell4 = np.log(data.Lambda4**2 / phys_pion**2)
    mean = ell4.mean()
    err =  ell4.std()
    syserr = None
    logging.info("{} l4: mean{} err{} syserr{}".format(label, mean, err, syserr))
    return (mean, err, syserr)

def sigma13(data, label=None):
    sigma = (data.B*(data.F_0)**2) #/2.
    s13 = sigma**(1./3.)
    mean = s13.mean()
    err =  s13.std()
    syserr = ((s13*0.01433447098)).mean()
    logging.info("{} sigma3: mean{} err{} syserr{}".format(label, mean, err, syserr))
    return (mean, err, syserr)

def plot_constants(axe, bootstrap_fit_files, options):

    values = {}
    errors = {}

    alldata = {}

    for f in bootstrap_fit_files:
        if "failed" in f.name:
            info.warn("included failed fit file {}".format(f.name))
        name, d = read_data(f)
        alldata[name] = d

    if "mpi_xi_inverse_NLO" in alldata.keys() and "fpi_xi_inverse_NLO" in alldata.keys():
        logging.info("merging mpi nlo with fpi nlo fits")
        alldata["fpi_mpi_xi_inverse_NLO"] =  pd.concat([alldata["mpi_xi_inverse_NLO"], alldata["fpi_xi_inverse_NLO"]], axis=1, join='inner')
        del alldata["mpi_xi_inverse_NLO"]
        del alldata["fpi_xi_inverse_NLO"]


    plots = []
    for k in sorted(alldata.keys()):
        data = alldata[k]
        if options.constant == "Lambda4":
            xdata, staterr, syserr = Lambda4(data, label=k)
        elif options.constant == "Lambda3":
            xdata, staterr, syserr = Lambda3(data, label=k)
        elif options.constant == "l4":
            xdata, staterr, syserr = l4(data, label=k)
        elif options.constant == "l3":
            xdata, staterr, syserr = l3(data, label=k)
        elif options.constant == "sigma13":
            xdata, staterr, syserr = sigma13(data, label=k)
        elif options.constant == "f0":
            xdata, staterr, syserr = f0(data, label=k)
        else:
            logging.error("Not a valid constant selection")
            exit(-1)

        thisy = plotindex.next()
        # color = colors.next()
        if "x" in k:
            color = 'b'
            if "NNLO" in k:
                label = "NNLO x"
                mark='o'
            elif "NLO" in k:
                label = "NLO x"
                mark='.'
        if "xi" in k or "XI" in k:
            color = 'r'
            if "NNLO" in k:
                label = "NNLO $\\xi$"
                mark='o'
            elif "NLO" in k:
                label = "NLO $\\xi$"
                mark='.'


        plotsettings = dict(linestyle="none", c=color, marker=mark,
                            label=label, ms=12, elinewidth=4,
                            capsize=8, capthick=2, mec=color, mew=2,
                            aa=True, mfc=color, fmt='o', ecolor=color)

        if syserr > 0:
            axe.errorbar(y=thisy, x=xdata, xerr=np.sqrt(staterr**2+ (syserr)**2), **dict(plotsettings, label=None))
        p = axe.errorbar(y=thisy, x=xdata, xerr=staterr, **plotsettings)
        plots.append(p)

    return plots

def otherplot(axe, x, xerr, label, labelx, plotsettings):
    y = plotindex.next()
    mark_gen = mark_generator()
    mark = mark_gen.next()
    axe.errorbar(y=y, x=x, xerr=xerr, marker=mark, **plotsettings)
    axe.annotate(label, xy=(labelx,y), fontsize=30)

def flagplot(axe, x, xerr, labelx, plotsettings):
    y = plotindex.next()
    axe.errorbar(y=y, x=x, xerr=xerr, **plotsettings)
    axe.annotate("FLAG", xy=(labelx,y), fontsize=30)
    axe.axvspan(x-xerr, x+xerr, facecolor='0.5', alpha=0.5)


def add_others(axe, options):
    divlinex = plotindex.next()
    axe.axhline(y=divlinex, color="k")
    color = 'k'
    flagplotsettings = dict(linestyle="none", c=color, marker="x",
                            ms=12, elinewidth=4,
                            capsize=8, capthick=2, mec=color, mew=2,
                            aa=True, mfc=color, fmt='o', ecolor=color)
    color = 'g'
    plotsettings = dict(linestyle="none", c=color,
                            ms=12, elinewidth=4,
                            capsize=8, capthick=2, mec=color, mew=2,
                            aa=True, mfc=color, fmt='o', ecolor=color)

    # if options.constant == "Lambda4":
    #     return axe.errorbar(y=plotindex.next(), x=LAMBDA4, xerr=LAMBDA4_err, **plotsettings)
    # if options.constant == "Lambda3":
    #     return axe.errorbar(y=plotindex.next(), x=LAMBDA3, xerr=LAMBDA3_err, **plotsettings)
    if options.constant == "l4":
        # axe.set_xlim(3,8)
        xlabel=5.2
        flagplot(axe,  4.10, 0.45,xlabel, flagplotsettings)
        axe.axvspan(4.10-0.45, 4.10+0.45, facecolor='0.5', alpha=0.5)
        otherplot(axe,  4.02, 0.253, "RBC/UKQCD15E",xlabel, plotsettings)
        otherplot(axe,  4.113, 0.59, "RBC/UKQCD14B",xlabel, plotsettings)
        otherplot(axe,  3.8, 0.44, "BMW13",xlabel, plotsettings)
        otherplot(axe,  3.99, 0.18, "RBC/UKQCD 12",xlabel, plotsettings)
        otherplot(axe,  4.03, 0.16, "Borsanyi 12",xlabel, plotsettings)
        otherplot(axe,  3.98, 0.32, "MILC 10A",xlabel, plotsettings)
        otherplot(axe,  4.30, 0.51, "NPLQCD 11",xlabel, plotsettings)
    if options.constant == "l3":
        xlabel=4.0
        # axe.set_xlim(0,6)
        flagplot(axe,  2.81, 0.64,xlabel, flagplotsettings)
        axe.axvspan(2.81-0.64, 2.81+0.64, facecolor='0.5', alpha=0.5)
        otherplot(axe,  2.81, 0.488, "RBC/UKQCD15E",xlabel, plotsettings)
        otherplot(axe,  2.73, 0.13, "RBC/UKQCD14B",xlabel, plotsettings)
        otherplot(axe,  2.5, 0.64, "BMW13",xlabel, plotsettings)
        otherplot(axe,  2.91, 0.24, "RBC/UKQCD 12",xlabel, plotsettings)
        otherplot(axe,  3.16, 0.30, "Borsanyi 12",xlabel, plotsettings)
        # otherplot(axe,  2.85, 0.81, "MILC 10A",xlabel, plotsettings)
        # otherplot(axe,  4.04, 0.40, "NPLQCD 11",xlabel, plotsettings)
    if options.constant == "sigma13":
        xlabel=290
        # axe.set_xlim(150,350)
        flagplot(axe,  274.0, 3.0,xlabel, flagplotsettings)
        axe.axvspan(274.0-3.0, 274.0+3.0, facecolor='0.5', alpha=0.5)
        otherplot(axe,  274.2, 4.88, "RBC/UKQCD15E",xlabel, plotsettings)
        otherplot(axe,  275.9, 2.15, "RBC/UKQCD14B",xlabel, plotsettings)
        otherplot(axe,  271.0, 4.0, "BMW13",xlabel, plotsettings)
        otherplot(axe,  281.5, 5.2, "MILC 10A",xlabel, plotsettings)
        otherplot(axe,  272.3, 1.84, "Borsanyi 12",xlabel, plotsettings)
        # otherplot(axe,  234.3, 17.4, "JLQCD/TWQCD 10",xlabel, plotsettings)
    if options.constant == "f0":
        xlabel=90
        # axe.set_ylim(70,100)
        # otherplot(axe,  271.0, 15.0, flagplotsettings)
        otherplot(axe,  88.0, 1.4, "BMW13",xlabel, plotsettings)
        otherplot(axe,  87.5, 1.0, "MILC 10A",xlabel, plotsettings)
        otherplot(axe,  86.78, 0.25, "Borsanyi 12",xlabel, plotsettings)
        # otherplot(axe,  234.3, 17.4, "JLQCD/TWQCD 10",xlabel, plotsettings)

    # divlinex = plotindex.next()
    # axe.axhline(y=divlinex, color="k")
    # axe.text(divlinex+0.5, 0.5, 'This work', fontsize=80)

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

    if args.output_stub is not None:
        root = logging.getLogger()
        errfilename = args.output_stub+".err"
        errfilehandler = logging.FileHandler(errfilename, delay=True)
        errfilehandler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        errfilehandler.setFormatter(formatter)
        root.addHandler(errfilehandler)
        logfilename = args.output_stub+".log"
        logfilehandler = logging.FileHandler(logfilename, delay=True)
        logfilehandler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        logfilehandler.setFormatter(formatter)
        root.addHandler(logfilehandler)



    fig, axe = plt.subplots(1)
    axe.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    plots = []
    plots.extend(plot_constants(axe, args.files, args))
    add_others(axe, args)

    fontsettings = dict(fontweight='bold', fontsize=50)

    axe.set_title("${}$".format(format_parameters(args.constant)), **fontsettings)
    axe.set_xlabel("${}$".format(format_parameters(args.constant)), **fontsettings)

    axe.tick_params(axis='x', which='major', labelsize=40)
    axe.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off') # labels along the bottom edge are off

    plots.reverse()
    axe.legend(handles=plots, loc=4, fontsize=30, numpoints=1)
    plt.ylim(0, plotindex.next())
    print("xlim:", axe.get_xlim())
    print("ylim:", axe.get_ylim())
    xlim = axe.get_xlim()
    plt.xlim(xlim[0], xlim[1]+(0.9*(xlim[1] - xlim[0])))

    if(args.output_stub):
        fig.set_size_inches(9.5, 12.5)
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

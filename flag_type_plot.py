#!/usr/bin/env python2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib as mpl
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

from commonplotlib.auto_key import auto_key

from plot_decay_constants import auto_fit_range
from add_chiral_fits import format_parameters

first = True

colors = list(reversed(['b', 'r', 'k', 'm', 'c', 'y']))*10

def mark_generator():
    for i in ["D", "^", "<", ">", "v", "x", "p", "8"]*2:
        yield i

def number_generator():
    for i in range(-1,-100,-1):
        yield i

plotindex = number_generator()

summary_lines = []

def add_this(axe, thiswork):

    color = colors.pop()
    label = "This work"
    for i in thiswork:
        value, err1, err2 = i
        plotsettings = dict(linestyle="none", c=color, marker="o",
                            ms=18, elinewidth=4,
                            capsize=10, capthick=5, mec=color, mew=2,
                            aa=True, mfc=color, fmt='o', ecolor=color)
        yloc = plotindex.next()
        plt.errorbar(x=value, y=yloc, xerr=err1, label=label ,**plotsettings)
        plt.errorbar(x=value, y=yloc, xerr=np.sqrt(err2**2 + err1**2), label="" ,**plotsettings)
        label=""


def add_flag(axe, flagfile):

    color = colors.pop()
    marks = mark_generator()
    for i in flagfile:
        plotsettings = dict(linestyle="none", c=color, marker=marks.next(),
                            ms=18, elinewidth=4,
                            capsize=10, capthick=5, mec=color, mew=2,
                            aa=True, mfc=color, fmt='o', ecolor=color)
        if i.startswith('#'):
            title = i.strip()[1:]
            continue
        name, value, err = i.split(",")
        value = float(value)
        err = float(err)
        yloc = plotindex.next()
        plt.errorbar(x=value, y=yloc, xerr=err, label=name, **plotsettings )
        if name.startswith("FLAG"):
            mid = value
            lower = value - err
            upper = value + err
            axe.axhline(y=yloc+0.5, linewidth=2, color='k', ls="dotted")
            axe.axhline(y=yloc-0.5, linewidth=2, color='k', ls="dotted")
            axe.axvspan(lower, upper, alpha=0.3, color='red')
            color = colors.pop()
        else:
            lower = min(lower, value-err)
            upper = max(upper, value+err)

    dist = max([mid-lower, upper-mid])
    plt.xlim(mid-dist*1.8, mid+dist*1.5)

def plot_flag(options):

    fig, axe = plt.subplots(1)

    plots = []
    add_this(axe, options.thiswork)
    add_flag(axe, options.flagfile)

    fontsettings = dict(fontweight='bold', fontsize=50)
    axe.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off') # labels along the bottom edge are off

    # axe.set_title("${}$".format(format_parameters(args.constant)), **fontsettings)
    # axe.set_ylabel("${}$".format(format_parameters(args.constant)), **fontsettings)

    # axe.tick_params(axis='y', which='major', labelsize=40)


    axe.legend(loc=2, fontsize=35, numpoints=1)
    plt.ylim(plotindex.next()-1, 0)

    options.flagfile.seek(0)
    label = options.flagfile.readline().strip("# \n")

    axe.tick_params(axis='x', which='major', labelsize=30)
    axe.set_xlabel(label, labelpad=20, **fontsettings)

    if(args.output_stub):
        # fig.set_size_inches(26.5, 9.5)
        width = 18.0
        fig.set_size_inches(width, width*1.2)
        fig.tight_layout()
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







if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="convert constants into other formats")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('flagfile', metavar='f', type=argparse.FileType('r'),
                        help='file with flag result ')
    parser.add_argument("--thiswork", required=False, nargs=3, metavar=("val", "err1", "err2"), type=float,
                        action='append', help="Add our data", default=None)
    parser.add_argument("--png", action="store_true",
                        help="save as png")
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


    plot_flag(args)

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

import commonplotlib.plot_helpers

from data_params import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from data_params import all_same_beta, all_same_heavy, all_same_flavor
from physical_values import phys_pion, phys_kaon
from commonplotlib.auto_key import auto_key


def read_bootstraps(f, options):


    df = pd.read_csv(f,comment='#', names=["config", "mass", "amp1", "amp2"])

    return df["mass"]


def plot_land_scape(options):
    fontsettings = dict(fontsize=40)
    fig, axe = plt.subplots(1)

    hbar_c = 197.327
    plots = []
    for f in options.files:
        dp = data_params(f)
        color = "r"
        if dp.s_mass == 0.03:
            continue
        if dp.s_mass == 0.018:
            continue
        if dp.ud_mass == 0.0035:
            if dp.latsize == "32x64x12":
                continue
            color='b'
        if dp.latsize == "64x128x08":
            color='m'
        data = read_bootstraps(f, options)
        average = np.mean(data)
        y = scale[dp.beta]*average
        # print data
        # print average
        # print dp
        # print dp.beta
        # print scale[dp.beta]

        # print 1.0/scale[dp.beta]
        a = hbar_c/scale[dp.beta]
        p = plt.scatter(a,y, label="{}".format(dp), s=400, color=color)
        plots.append(p)

    # plt.legend(handles=sorted(plots), loc=0, **fontsettings)
    plt.scatter(0, phys_pion, marker="s", s=400, color="m")
    plt.annotate('physical', xy=(0, phys_pion), xytext=(0.01, 140),
            arrowprops=dict(facecolor='black', shrink=0.05), **fontsettings
            )
    x1 = np.linspace(0,0.15)
    x2 = np.linspace(0.1,0.15)
    x3 = np.linspace(0.15,2.0)
    v = np.linspace(0,800)
    plt.fill_between(x1,500,1000, facecolor="c", alpha=0.4, lw=0, zorder=-10)
    plt.fill_between(x1,300,500, facecolor="c", alpha=0.1, lw=0, zorder=-10)

    # plt.fill_between(x2,0,300, facecolor="c", alpha=0.4, lw=0, zorder=-10)
    plt.fill_between(x2,0,300, facecolor="c", alpha=0.1, lw=0, zorder=-10)
    plt.fill_between(x3,0,1000, facecolor="c", alpha=0.4, lw=0, zorder=-10)
    plt.xlim(0,0.165)
    plt.ylim(0,700)

    plt.xticks(np.arange(0, 0.16, 0.05))
    plt.yticks(np.arange(0, 700, 200))
    plt.ylabel("$M_\pi [MeV]$", **fontsettings)
    plt.xlabel("a[fm]", **fontsettings)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # plt.show()


    if(options.output_stub):

        fig.set_size_inches(15.5, 10.5)
        file_extension = ".png"
        if options.eps:
            file_extension = ".eps"
        if options.pdf:
            file_extension = ".pdf"
        filename = options.output_stub + file_extension
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename)
        return

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="average data files")

    axis_choices = ["mud", "mud_s", "mpi", "mpisqr", "2mksqr-mpisqr"]
    legend_choices = ["betaLs", "betaL", "heavy", "smearing", "flavor", "strange"]

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-e", "--eps", action="store_true",
                        help="save as eps not png")
    parser.add_argument("--pdf", action="store_true",
                        help="save as pdf not png")
    parser.add_argument("-b", "--box", action="store_true",
                        help="max boxplots instead")
    parser.add_argument("-c", "--scatter", action="store_true",
                        help="make a scatter plot instead")
    parser.add_argument("-y", "--yrange", type=float, required=False, nargs=2,
                        help="set the yrange of the plot", default=None)
    parser.add_argument("-x", "--xrange", type=float, required=False, nargs=2,
                        help="set the xrange of the plot", default=None)
    parser.add_argument("--xaxis", required=False, choices=axis_choices,
                        help="what to set on the xaxis", default="mud")
    parser.add_argument("--legend_mode", required=False, choices=legend_choices,
                        help="what to use for the legend", default="betaLs")
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument("-t", "--title", type=str, required=False,
                        help="plot title", default="decay constants")
    parser.add_argument("--ylabel", type=str, required=False,
                        help="ylabel", default=None)
    parser.add_argument("--xlabel", type=str, required=False,
                        help="xlabel", default=None)
    parser.add_argument("-s", "--scale", action="store_true",
                        help="scale the values")
    parser.add_argument("-ss", "--scalesquared", action="store_true",
                        help="scale the values squared")
    parser.add_argument("-p", "--physical", type=float, nargs=2,
                        help="add physical point")
    parser.add_argument("-I", "--interpolate", type=argparse.FileType('r'), required=False,
                        help="add interpolated lines")
    parser.add_argument("--chiral_fit_file", type=argparse.FileType('r'), required=False,
                        help="add chiral interpolated lines")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    for f in args.files:
        if not os.path.isfile(f):
            raise argparse.ArgumentTypeError("Argument {} is not a valid file".format(f))

    logging.info("Ploting decay constants for: {}".format("\n".join(args.files)))

    if args.output_stub:
        outdir = os.path.dirname(args.output_stub)
        if not os.path.exists(outdir):
            logging.info("directory for output {} does not exist, atempting to create".format(outdir))
            if outdir is not "":
                os.makedirs(outdir)

    if args.scalesquared:
        args.scale = True


    plot_land_scape(args)

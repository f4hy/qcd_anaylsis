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


def determine_beta(files):
    logging.info("Determining beta")
    beta_filesnames = [re.search("_b([0-9]\.[0-9]*)_", f).group(1) for f in files]
    if allEqual(beta_filesnames):
        return beta_filesnames[0]
    else:
        raise RuntimeError("Not all betas are the same")


def allEqual(lst):
    return not lst or lst.count(lst[0]) == len(lst)


def plot_decay_constant(options):
    flavor_map = {"ud-ud": "\pi", "ud-s": "K", "s-s": "\eta", "heavy-ud": "Hl", "heavy-s": "Hs"}
    markers = ['o', "D", "^", "<", ">", "v", "x", "p", "8"]
    colors = ['b', 'r', 'k', 'm', 'c', 'y']
    flavor_color = {"\pi": 'b', "K": 'r', '\eta': 'm', "Hl": 'c', "Hs": 'y'}
    heavy_color = {"m0": 'b', "m1": 'r', 'm2': 'm'}

    beta = determine_beta(options.files)
    if beta == "4.47":
        heavy_masses = {"m0": 0.15, "m1": 0.206, "m2": 0.250}
        circle = mlines.Line2D([], [], color='black', marker='o', mfc='white', mew=3, lw=0,
                               markersize=15, label='$m_s=0.015$')
        cross = mlines.Line2D([], [], color='black', marker='D', lw=0,
                              markersize=15, label='$m_s=0.025$')
        s_mass_cutoff = 0.02
        scale = 4600
    if beta == "4.35":
        heavy_masses = {"m0": 0.12, "m1": 0.24, "m2": 0.36}
        circle = mlines.Line2D([], [], color='black', marker='o', mfc='white', mew=3, lw=0,
                               markersize=15, label='$m_s=0.025$')
        cross = mlines.Line2D([], [], color='black', marker='D', lw=0,
                              markersize=15, label='$m_s=0.018$')
        s_mass_cutoff = 0.02
        scale = 3600
    if beta == "4.17":
        heavy_masses = {"m0": 0.2, "m1": 0.4, "m2": 0.6}
        circle = mlines.Line2D([], [], color='black', marker='o', mfc='white', mew=3, lw=0,
                               markersize=15, label='$m_s=0.04$')
        cross = mlines.Line2D([], [], color='black', marker='D', lw=0,
                              markersize=15, label='$m_s=0.03$')
        s_mass_cutoff = 0.035
        scale = 2450

    fontsettings = dict(fontsize=30)

    flavor_patches = [mpatches.Patch(color=c, label='${}$'.format(l)) for l,c in flavor_color.iteritems() ]
    heavy_patches = [mpatches.Patch(color=c, label='${}$'.format(l)) for l,c in flavor_color.iteritems() ]

    legend_handles = [circle, cross]
    added_handles = []

    data = {}
    index = 0
    fig, axe = plt.subplots(1)
    for f in options.files:
        ud_mass = float(re.search("mud([0-9]\.[0-9]*)_", f).group(1))
        s_mass = float(re.search("ms([0-9]\.[0-9]*)", f).group(1))
        flavor = flavor_map[re.search("_([a-z][a-z]*-[a-z][a-z]??).out", f).group(1)]
        heavyness = re.search("_([a-z][a-z0-9])_", f).group(1)
        label = "$f_{}$ s{}".format(flavor, s_mass)
        with open(f) as datafile:
            x,y,e = [float(i.strip()) for i in datafile.read().split(",")]
        mark = markers[index % len(markers)]
        color = colors[index % len(colors)]

        if heavyness == "ll":
            color = flavor_color[flavor]
            if flavor not in added_handles:
                legend_handles.append(mpatches.Patch(color=color, label='${}$'.format(flavor)))
                added_handles.append(flavor)
        else:
            color = heavy_color[heavyness]
            if heavyness not in added_handles:
                legend_handles.append(mpatches.Patch(color=color, label='${}$'.format(heavy_masses[heavyness])))
                added_handles.append(heavyness)

        if "48x96x12" in f:
            color = 'g'
            latsize = re.search("_([0-9]*x[0-9]*x[0-9]*)_", f).group(1)
            if latsize not in added_handles:
                legend_handles.append(mpatches.Patch(color=color, label=latsize))
                added_handles.append(latsize)

        mark = 'o'
        mfc='white'
        if s_mass < s_mass_cutoff:
            mark = 'D'
            mfc = color
        plotsettings = dict(linestyle="none", c=color, marker=mark, label=label, ms=8, elinewidth=3, capsize=8,
                            capthick=2, mec=color, mew=3, aa=True, mfc=mfc, fmt='o')
        index+=1
        logging.info("plotting {} {} {}".format(x,y,e))
        axe.errorbar(x, y, yerr=e, zorder=0, **plotsettings)

    if options.xrange:
        logging.info("setting x range to {}".format(options.xrange))
        plt.xlim(options.xrange)

    if options.yrange:
        logging.info("setting y range to {}".format(options.yrange))
        plt.ylim(options.yrange)

    if options.title:
        axe.set_title(options.title, **fontsettings)

    axe.set_xlabel("$m_{ud}+m_{res}$", **fontsettings)
    axe.set_ylabel("lattice units", **fontsettings)
    axe.tick_params(axis='both', which='major', labelsize=20)

    if options.yrange:
        ax2 = axe.twinx()
        ax2.set_ylabel('MeV', **fontsettings)
        ax2.set_ylim(np.array(options.yrange) * scale)
        ax2.tick_params(axis='both', which='major', labelsize=20)

    leg = axe.legend(handles=legend_handles, loc=0, **fontsettings )
    if(options.output_stub):
        fig.set_size_inches(18.5, 10.5)
        if args.eps:
            logging.info("Saving plot to {}".format(options.output_stub+".eps"))
            plt.savefig(options.output_stub+".eps")
        else:
            logging.info("Saving plot to {}".format(options.output_stub+".png"))
            plt.savefig(options.output_stub+".png", dpi=200)
        return

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="average data files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-e", "--eps", action="store_true",
                        help="save as eps not png")
    parser.add_argument("-y", "--yrange", type=float, required=False, nargs=2,
                        help="set the yrange of the plot", default=None)
    parser.add_argument("-x", "--xrange", type=float, required=False, nargs=2,
                        help="set the xrange of the plot", default=None)
    parser.add_argument("-t", "--title", type=str, required=False,
                        help="plot title", default="decay constants")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Computing decay constants for: {}".format("\n".join(args.files)))

    if args.output_stub:
        outdir = os.path.dirname(args.output_stub)
        if not os.path.exists(outdir):
            logging.info("directory for output {} does not exist, atempting to create".format(outdir))
            if outdir is not "":
                os.makedirs(outdir)


    plot_decay_constant(args)

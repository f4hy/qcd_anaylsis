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

import plot_helpers

from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from ensamble_info import all_same_beta, all_same_heavy, all_same_flavor



def round5(x):
    return int(5 * np.around(x/5.0))

def round(x,y):
    log = np.log10(y-x)
    scale = 1
    if log < 0:
        digits = -np.floor(log)
        scale = 10**(digits+1)
        return round5(np.floor(x*scale))/scale, round5(np.ceil(y*scale))/scale
    return x,y


def auto_fit_range(minval, maxval, zero=False, buff=0.4):
    spread = maxval - minval
    if zero:
        fitrange = (0, maxval+spread*buff)
    elif spread == 0:
        fitrange = minval-(minval*0.1), maxval+(maxval*0.1)
    else:
        fitrange = round(minval-spread*buff, maxval+spread*buff)
    logging.info("setting range to {}".format(fitrange))
    return fitrange


def xvalues(xaxis_type, data_properties, options):
    logging.info("using xaxis type {}".format(xaxis_type))

    if options.scale:
        s = scale[data_properties.beta]
    else:
        s = 1.0

    if xaxis_type == "mud":
        residual = residual_mass(data_properties.ud_mass, data_properties.s_mass)
        return pd.Series(s*(data_properties.ud_mass + residual))

    if xaxis_type == "mud_s":
        residual = residual_mass(data_properties.ud_mass, data_properties.s_mass)
        return pd.Serites(s*(data_properties.ud_mass + residual + data_properties.s_mass + residual))

    if xaxis_type == "mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", options.fitdata)
        return (s*pionmass)**2 #*pionmass

    if xaxis_type == "2mksqr-mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", options.fitdata)
        kaonmass = read_fit_mass(data_properties, "ud-s", options.fitdata)
        return 2.0*(s*kaonmass)**2 - (s*pionmass)**2




legend_handles = []
added_handles = []
summary_lines = []
s_mass_marks = {}
markers = ['o', "D", "^", "<", ">", "v", "x", "p", "8", 'o', "D"]
colors = ['b', 'r', 'k', 'm', 'c', 'y', 'b', 'r', 'k', 'm', 'c', 'y']

beta_colors = {"4.17": 'b', "4.35": 'r', "4.47": 'k'}

heavy_color = {"m0": 'b', "m1": 'r', 'm2': 'm'}
heavy_colors = {}

smearing_colors = {}

def strange_legend(s_mass):
    if s_mass not in added_handles:
        s_mass_marks[s_mass] = markers.pop()
        mark = s_mass_marks[s_mass]
        smass_leg = mlines.Line2D([], [], color='black', marker=s_mass_marks[s_mass], mfc='white', mew=3, lw=0,
                                markersize=18, label='$m_s={}$'.format(s_mass))
        #legend_handles.append(smass_leg)
        added_handles.append(s_mass)
    else:
        mark = s_mass_marks[s_mass]
    return mark

flavor_color = {"\pi": 'b', "K": 'r', '\eta': 'm', "Hl": 'c', "Hs": 'y', "HH": 'g'}


def colors_and_legend(data_properties, one_beta, one_flavor):

    p = data_properties


    # if heavyness != "ll":
    #     if heavymass not in added_handles:
    #         print heavy_colors
    #         heavy_colors[heavymass] = colors.pop()
    #         color = heavy_colors[heavymass]
    #         legend_handles.append(mpatches.Patch(color=color, label='${}$'.format(heavymass)))
    #         added_handles.append(heavymass)
    #     else:
    #         color = heavy_colors[heavymass]
    #     return color


    if one_beta and one_flavor:
        logging.info("Only one beta nd one flavor given, using smearing")
        if p.smearing not in added_handles:
            smearing_colors[p.smearing] = colors.pop()
            color = smearing_colors[p.smearing]
            legend_handles.append(mpatches.Patch(color=color, label='${}$'.format(p.smearing)))
            added_handles.append(p.smearing)
        else:
            color = smearing_colors[p.smearing]
        return color



    if one_beta:
        color = flavor_color[p.flavor]
        if p.flavor not in added_handles:
            legend_handles.append(mpatches.Patch(color=color, label='${}$'.format(p.flavor)))
            added_handles.append(p.flavor)
        return color

    color = beta_colors[p.beta]
    if p.beta not in added_handles:
        mylabel = r'$\beta = {}$'.format(p.beta)
        legend_handles.append(mpatches.Patch(color=beta_colors[p.beta], label=mylabel))
        added_handles.append(p.beta)
    return color



def plot_decay_constant(options):


    #plt.rc('text', usetex=True)

    xmax = ymax = -10000
    ymin = 100000000

    fontsettings = dict(fontsize=30)

    flavor_patches = [mpatches.Patch(color=c, label='${}$'.format(l)) for l,c in flavor_color.iteritems() ]
    heavy_patches = [mpatches.Patch(color=c, label='${}$'.format(l)) for l,c in flavor_color.iteritems() ]

    has_shifted = any("msshifted" in f for f in options.files)


    one_beta = all_same_beta(options.files)
    one_flavor = all_same_flavor(options.files)
    logging.info("one_beta: {}, one_flavor: {}".format(one_beta, one_flavor))

    data = {}
    index = 0
    fig, axe = plt.subplots(1)


    for f in options.files:

        p = data_params(f)

        label = "$f_{}$ s{}".format(p.flavor, p.s_mass)

        df = pd.read_csv(f,comment='#', names=["decay"])



        mark = markers[index % len(markers)]
        # color = colors[index % len(colors)]

        mfc = 'white'

        mark = strange_legend(p.s_mass)

        color = colors_and_legend(p, one_beta, one_flavor)

        # if "48x96x12" in f:
        #     logging.info("48x96x12!!!!")
        #     color = 'g'
        #     latsize = re.search("_([0-9]*x[0-9]*x[0-9]*)_", f).group(1)
            # if latsize not in added_handles:
            #     legend_handles.append(mpatches.Patch(color=color, label=r"$\beta = 4.17 48x96$"))
            #     added_handles.append(latsize)

        alpha = 1.0
        if "32x64" in f and p.ud_mass < 0.004:
            alpha = 0.3
            color="#9999FF"

        xs = xvalues(options.xaxis, p, options)
        x = xs.median()
        xerr = xs.std()
        xerr = plot_helpers.error(xs)
        y = float(df.median())
        e = float(df.std().values)
        e = plot_helpers.error(df.values)

        summary_lines.append("{}, {}, {}\n".format(p, y, df.std().values))

        scalepower = 1.0
        if options.scalesquared:
            scalepower = 2.0

        if has_shifted and p.s_mass != "shifted":
            alpha = 0.3


        plotsettings = dict(linestyle="none", c=color, marker=mark, label=label, ms=8, elinewidth=3, capsize=8,
                            capthick=2, mec=color, mew=3, aa=True, mfc=mfc, fmt='o', ecolor=color, alpha=alpha)
        index+=1
        logging.info("plotting {} {} {}".format(x,y,e))
        if options.scale:
            sc = scale[p.beta]**scalepower
            scaled_err = plot_helpers.error(df.values*sc)

            if options.box:
                b = axe.boxplot(df["decay"]*sc, positions=[x], widths=[0.001*sc], patch_artist=True)
            elif options.scatter:
                axe.scatter(xs, df["decay"]*sc, c=color)
            else:
                axe.errorbar(x, y*sc, yerr=scaled_err, xerr=xerr, zorder=0, **plotsettings)
            ymax = max(ymax,y*sc)
            ymin = min(ymin,y*sc)
            xmax = max(xmax,x)

        else:
            if options.box:
                b = axe.boxplot(df["decay"], positions=[x], widths=[0.001], patch_artist=True)
            else:
                axe.errorbar(x, y, yerr=e, xerr=xerr, zorder=0, **plotsettings)
            ymax = max(ymax,y)
            ymin = min(ymin,y)
            xmax = max(xmax,x)

        if options.box:
            b["boxes"][0].set_color(color)
            b["boxes"][0].set_facecolor(color)
            b["medians"][0].set_color('k')
            plt.setp(b["fliers"], visible=False)


    if options.physical:
        x_physicals = {"mud": 2.2, "mud_s": 97.2, "mpisqr": 138.0**2, "2mksqr-mpisqr": 2*(497.6**2)-138.0**2}
        y, err = options.physical
        if options.scale:
            x = x_physicals[options.xaxis]
        else:
            x = 0.001
        physplot = axe.errorbar(x, y, yerr=err, marker="o", ecolor="k", color="k", label="physical",
                                ms=15, elinewidth=3, capsize=1, capthick=2, mec=color, mew=3, mfc='m')
        legend_handles.append(physplot)
        ymax = max(ymax,y)
        ymin = min(ymin,y)

    if options.xrange:
        logging.info("setting x range to {}".format(options.xrange))
        plt.xlim(options.xrange)
    else:
        logging.info("auto setting x range")
        plt.xlim(auto_fit_range(0, xmax, zero=True))


    if options.yrange:
        logging.info("setting y range to {}".format(options.yrange))
        plt.ylim(options.yrange)
    else:
        logging.info("auto setting y range")
        if options.box:
            plt.ylim(auto_fit_range(ymin, ymax, buff=3.5))
        else:
            plt.ylim(auto_fit_range(ymin, ymax))


    if options.title:
        axe.set_title(options.title, **fontsettings)

    #axe.set_xlabel("$m_{%s}+m_{res}+m_{%s}+m_{res}$" % (rawflavor.split("-")[0], rawflavor.split("-")[1]), **fontsettings)

    xlabel = {"mud": u"$m_{l}+m_{res}$", "mud_s": u"$m_{l}+m_s+2m_{res}$", "mpi": u"$m_{\pi}$",
              "mpisqr": u"$m^2_{\pi}$", "2mksqr-mpisqr": u"$2m^2_{K}-m^2_{\pi}$" }

    axe.set_xlabel(xlabel[options.xaxis], **fontsettings)

    if options.scale:
        if options.scalesquared:
            axe.set_ylabel("MeV^2", **fontsettings)
        else:
            axe.set_ylabel("MeV", **fontsettings)
    else:
        axe.set_ylabel("lattice units", **fontsettings)

    axe.tick_params(axis='both', which='major', labelsize=20)


    if not options.box:
        leg = axe.legend(handles=sorted(legend_handles), loc=0, **fontsettings )
    if(options.output_stub):
        summaryfilename = options.output_stub + ".txt"
        logging.info("Writting summary to {}".format(summaryfilename))
        with open(summaryfilename, 'w') as summaryfile:
            for i in summary_lines:
                summaryfile.write(i)

        fig.set_size_inches(18.5, 10.5)
        if args.eps:
            logging.info("Saving plot to {}".format(options.output_stub+".eps"))
            plt.savefig(options.output_stub+".eps")
        else:
            logging.info("Saving plot to {}".format(options.output_stub+".png"))
            plt.savefig(options.output_stub+".png", dpi=200)
        return

    print "".join(summary_lines)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="average data files")

    axis_choices = ["mud", "mud_s", "mpi", "mpisqr", "2mksqr-mpisqr"]

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-e", "--eps", action="store_true",
                        help="save as eps not png")
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
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument("-t", "--title", type=str, required=False,
                        help="plot title", default="decay constants")
    parser.add_argument("-s", "--scale", action="store_true",
                        help="scale the values")
    parser.add_argument("-ss", "--scalesquared", action="store_true",
                        help="scale the values squared")
    parser.add_argument("-p", "--physical", type=float, nargs=2,
                        help="add physical point")
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


    plot_decay_constant(args)

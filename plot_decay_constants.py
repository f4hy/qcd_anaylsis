#!/usr/bin/env python2
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import logging
import argparse
import os


import numpy as np

from residualmasses import residual_mass, residual_mass_errors

from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from ensamble_info import all_same_beta, all_same_heavy, all_same_flavor
from ensamble_info import phys_pion, phys_kaon, phys_mq, phys_Fpi, phys_FD, phys_FDs, phys_D, phys_Ds
from ensamble_info import phys_eta, phys_etac, phys_FK, phys_mhq
from ensamble_info import Zs, Zv

from ensemble_data import ensemble_data, NoStrangeInterp

from auto_key import auto_key

from add_chiral_fits import add_chiral_fit

from get_data import get_data

from itertools import cycle

def round5(x):
    return int(5 * np.around(x/5.0))


def round(x, y):
    log = np.log10(y-x)
    scale = 1
    if log < 0:
        digits = -np.floor(log)
        scale = 10**(digits+1)
        return round5(np.floor(x*scale))/scale, round5(np.ceil(y*scale))/scale
    return x, y


def auto_fit_range(minval, maxval, zero=False, buff=0.5):
    spread = maxval - minval
    if zero:
        fitrange = (0, maxval+spread*buff)
    elif spread == 0:
        fitrange = minval-(minval*0.1), maxval+(maxval*0.1)
    else:
        fitrange = round(minval-spread*buff, maxval+spread*buff)
    logging.info("setting range to {}".format(fitrange))
    return fitrange


legend_handles = []
added_handles = []
summary_lines = []

phys_marks = cycle('x+s')

def colors_and_legend(data_properties, legend_mode="betaLs"):

    p = data_properties

    if legend_mode == "custom":
        colors = {"4.17": 'b', "4.35": 'r', "4.47": 'm'}
        marks = {"4.17": 'o', "4.35": 's', "4.47": 'd'}
        labels = {"4.17": r'$a^{-1} = 2.45$GeV', "4.35": r'$a^{-1} = 3.61$GeV', "4.47": r'$a^{-1} = 4.50$GeV'}
        color = colors[p.beta]
        mark = marks[p.beta]
        legend_label = labels[p.beta]
        mfc = None
        nolegends = [0.03, 0.018]
        if float(p.s_mass) in nolegends:
            mfc = "white"
            legend_label = None

    if legend_mode == "betaLs":
        color, mark, mfc = auto_key((p.beta, p.s_mass, p.latsize))
        legend_label = r'$\beta = {}, L={}, m_s={}$'.format(p.beta, p.latsize[:2], p.s_mass)
        if p.s_mass =="shifted":
            legend_label = r'$\beta = {}, L={}$'.format(p.beta, p.latsize[:2])

    if legend_mode == "betaL":
        color, mark, mfc = auto_key((p.beta, p.s_mass))
        legend_label = r'$\beta = {}, L={}$'.format(p.beta, p.latsize[:2])

    if legend_mode == "strange":
        color, mark, mfc = auto_key(p.s_mass)
        legend_label = r'$ms={}$'.format(p.s_mass)

    if legend_mode == "heavy":
        color, mark, mfc = auto_key(p.heavymass)
        legend_label = r'$mh={}$'.format(p.heavymass)

    if legend_mode == "betaheavy":
        color, mark, mfc = auto_key((p.beta, p.heavymass))
        legend_label = r'$\beta = {}, mh={}$'.format(p.beta, p.heavymass)


    if legend_mode == "smearing":
        color, mark, mfc = auto_key(p.smearing)
        legend_label = r'$smearing={}$'.format(p.smearing)

    if legend_mode == "flavor":
        color, mark, mfc = auto_key(p.flavor)
        legend_label = r'$smearing={}$'.format(p.flavor)

    symbol = mpl.lines.Line2D([], [], color=color, mec=color, marker=mark, markersize=15,
                              linestyle="None", label=legend_label, mfc=mfc)
    #legend_handles.append(mpatches.Patch(color=beta_colors[p.beta], label=mylabel))
    handel = (color, mark, mfc)
    if handel not in added_handles:
        if legend_label is not None:
            added_handles.append(handel)
            legend_handles.append(symbol)

    return handel


def plot_decay_constant(options):

    xmax = ymax = -10000
    ymin = 100000000

    fontsettings = dict(fontsize=60)

    one_beta = all_same_beta(options.files)
    one_flavor = all_same_flavor(options.files)
    logging.info("one_beta: {}, one_flavor: {}".format(one_beta, one_flavor))

    fig, axe = plt.subplots(1)

    for f in options.files:

        p = data_params(f)

        ed = ensemble_data(p, interpstrange=options.interpstrange)

        try:
            y, yerr, ylabel, yphysical = get_data(ed, options.ydata, options)
            x, xerr, xlabel, xphysical = get_data(ed, options.xdata, options)
        except NoStrangeInterp as interperror:
            logging.warn("for {} found error {}".format(f, interperror))
            continue

        e = yerr

        label = "$f_{}$ s{}".format(p.flavor, p.s_mass)

        color, mark, mfc = colors_and_legend(p, options.legend_mode)

        summary_lines.append("{}, {}, {}\n".format(p, y, e))

        alpha = 1.0
        if "32x64" in f and p.ud_mass < 0.004:
            alpha = 0.6
            color = "#9999FF"
            #continue

        if options.mhcut and p.heavyq_mass > options.mhcut:
            alpha = 0.1

        plotsettings = dict(linestyle="none", c=color, marker=mark,
                            label=label, ms=15, elinewidth=4,
                            capsize=8, capthick=2, mec=color, mew=2,
                            aa=True, mfc=mfc, fmt='o', ecolor=color,
                            alpha=alpha)
        logging.info("plotting {} {} {}".format(x, y, e))

        if options.xerror is False:
            xerr = 0.0
        axe.errorbar(x, y, yerr=yerr, xerr=xerr, zorder=0, **plotsettings)
        ymax = max(ymax, y)
        ymin = min(ymin, y)
        xmax = max(xmax, x)

    if options.physical:
        logging.info("plotting physical {} {}".format(xphysical, yphysical))
        matchingkeys = set(xphysical.keys()) & set(yphysical.keys())

        if len(matchingkeys) > 1:
            physiter = [(k, xphysical[k], yphysical[k]) for k in matchingkeys ]
        else:
            physiter = [(k, xp, yphysical[k]) for k in yphysical.keys() for xl, xp in xphysical.iteritems()]
        for yl, xp, yp in physiter:
            pmark = phys_marks.next()
            physplot = axe.errorbar(xp, yp, yerr=0, marker=pmark,
                                    ecolor="k", color="k", label=yl,
                                    ms=15, elinewidth=3, capsize=1,
                                    capthick=2, mec='k', mew=3, mfc='k',
                                    zorder=100)
            symbol = mpl.lines.Line2D([], [], color="k", mec="k", marker=pmark, markersize=15, mew=3,
                                      linestyle="None", label=yl, mfc="k")
            legend_handles.append(symbol)
            ymax = max(ymax, yp)
            ymin = min(ymin, yp)

    if options.scalelines:
        if options.xdata.startswith("1/"):
            for i in scale.keys():
                physxplot = axe.axvline(1.0/scale[i], color=auto_key((i, None, None), check=False)[0], ls="--", lw=2, label=i)
        else:
            for i in scale.keys():
                physxplot = axe.axvline(scale[i], color=auto_key((i, None, None), check=False)[0], ls="--", lw=2, label=i)

    if options.physx:
        physxplot = axe.axvline(xphysical, color='k', ls="--", lw=2, label="physical point")
        legend_handles.append(physxplot)

    if options.addpoint:
        print options.addpoint
        px = float(options.addpoint[1])
        py = float(options.addpoint[2])
        physplot = axe.errorbar(px, py, yerr=0, marker="x",
                                ecolor="k", color="k", label=options.addpoint[0],
                                ms=15, elinewidth=3, capsize=1,
                                capthick=2, mec='k', mew=3, mfc='k',
                                zorder=100)
        symbol = mpl.lines.Line2D([], [], color="k", mec="k", marker="x", markersize=15, mew=3,
                                  linestyle="None", label=options.addpoint[0], mfc="k")
        legend_handles.append(symbol)


    if options.xrange:
        logging.info("setting x range to {}".format(options.xrange))
        xran = options.xrange
    else:
        logging.info("auto setting x range")
        xran = auto_fit_range(0, xmax, zero=True)

    plt.xlim(xran)

    if options.yrange:
        logging.info("setting y range to {}".format(options.yrange))
        plt.ylim(options.yrange)
    else:
        logging.info("auto setting y range")
        if options.box:
            plt.ylim(auto_fit_range(ymin, ymax, buff=5.5))
        else:
            plt.ylim(auto_fit_range(ymin, ymax, buff=0.5))

    if options.chiral_fit_file:
        del legend_handles[:]
        for i in options.chiral_fit_file:
            chiral_line = add_chiral_fit(axe, xran, i, options)
            legend_handles.extend(chiral_line)

    if options.xlabel:
        axe.set_xlabel(options.xlabel, labelpad=20, **fontsettings)
    else:
        axe.set_xlabel(xlabel, labelpad=20, **fontsettings)
        if "MeV^2" in xlabel:
            import matplotlib.ticker as ticker
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(1000**2)))
            axe.xaxis.set_major_formatter(ticks)
            axe.set_xlabel(xlabel.replace("MeV","GeV"), **fontsettings)
        if "1/MeV" in xlabel:
            import matplotlib.ticker as ticker
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(1000*x))
            axe.xaxis.set_major_formatter(ticks)
            axe.set_xlabel(xlabel.replace("1/MeV","1/GeV"), **fontsettings)


    if options.ylabel:
        if options.scale:
            axe.set_ylabel("{} [MeV]".format(options.ylabel), labelpad=10, **fontsettings)
        else:
            axe.set_ylabel("{}".format(options.ylabel), labelpad=10, **fontsettings)
    else:
        axe.set_ylabel("{}".format(ylabel), labelpad=10, **fontsettings)

    axe.tick_params(axis='both', which='major', labelsize=30)

    if options.title:
        fig.suptitle(options.title.replace("_", " "), **fontsettings)


    def legsort(i):
        return i.get_label()



    if not options.box:
        leg = axe.legend(handles=sorted(legend_handles, key=legsort), loc=options.legendloc,
                         fontsize=30, numpoints=1)
    if(options.output_stub):
        summaryfilename = options.output_stub + ".txt"
        logging.info("Writting summary to {}".format(summaryfilename))
        with open(summaryfilename, 'w') as summaryfile:
            for i in summary_lines:
                summaryfile.write(i)

        width = 20.0
        fig.set_size_inches(width, width*options.aspect)
        fig.tight_layout()
        #fig.set_size_inches(28.5, 20.5)
        file_extension = ".png"
        if options.eps:
            file_extension = ".eps"
        if options.pdf:
            file_extension = ".pdf"
        filename = options.output_stub + file_extension
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename)
        return

    logging.info("".join(summary_lines))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="average data files")

    axis_choices = ["mud", "mud_s", "mpi", "mpisqr", "2mksqr-mpisqr", "mpisqr/mq", "xi", "mq"]
    legend_choices = ["betaLs", "betaL", "heavy", "smearing", "flavor", "strange", "betaheavy", "custom"]

    extention_group = parser.add_mutually_exclusive_group()
    extention_group.add_argument("--png", action="store_true", default=True,
                                 help="save as eps not png")
    extention_group.add_argument("--eps", action="store_true",
                                 help="save as eps not png")
    extention_group.add_argument("--pdf", action="store_true",
                                 help="save as pdf not png")


    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("-b", "--box", action="store_true",
                        help="max boxplots instead")
    parser.add_argument("-c", "--scatter", action="store_true",
                        help="make a scatter plot instead")
    parser.add_argument("--scalelines", action="store_true",
                        help="plot lines indicating scale cutoff")
    parser.add_argument("-y", "--yrange", type=float, required=False, nargs=2,
                        help="set the yrange of the plot", default=None)
    parser.add_argument("-x", "--xrange", type=float, required=False, nargs=2,
                        help="set the xrange of the plot", default=None)
    parser.add_argument("--addpoint", required=False, nargs=3, metavar=("LABEL", "X", "Y"),
                        help="Add a point", default=None)
    parser.add_argument("--xaxis", required=False, choices=axis_choices,
                        help="what to set on the xaxis", default="mud")
    parser.add_argument("--legendloc", type=int, required=False, default=0,
                        help="location of legend")
    parser.add_argument("--legend_mode", required=False, choices=legend_choices,
                        help="what to use for the legend", default="betaLs")
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument("-t", "--title", type=str, required=False,
                        help="plot title", default="")
    parser.add_argument("--ylabel", type=str, required=False,
                        help="ylabel", default=None)
    parser.add_argument("--xlabel", type=str, required=False,
                        help="xlabel", default=None)
    parser.add_argument("--xerror", action="store_true",
                        help="plot x errors")
    parser.add_argument("-s", "--scale", action="store_true",
                        help="scale the values")
    parser.add_argument("-ss", "--scalesquared", action="store_true",
                        help="scale the values squared")
    parser.add_argument("--physical", action="store_true",
                        help="add physical point")
    parser.add_argument("--physx", action="store_true",
                        help="draw line at physical x")
    parser.add_argument("-I", "--interpolate", type=argparse.FileType('r'), required=False,
                        help="add interpolated lines")
    parser.add_argument("--chiral_fit_file", type=argparse.FileType('r'), required=False,
                        action='append', help="add chiral interpolated lines")
    parser.add_argument("--mpisqrbymq", action="store_true",
                        help="compute mpisqr divided by mq, strange edge case")
    parser.add_argument("--ydata", required=False, type=str,
                        help="which data to plot on the yaxis", default="mpi")
    parser.add_argument("--xdata", required=False, type=str,
                        help="what to use as the xaxis", default="mud")
    parser.add_argument("--aspect", type=float, default=1.0, required=False,
                        help="determine the plot aspect ratio")
    parser.add_argument("--interpstrange", action="store_true",
                        help="use interpoalted strange masses", default=None)
    parser.add_argument("--mhcut", type=float, default=None, required=False,
                        help="cut of mh to fade")



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
            logging.info("directory for output {} does not exist,"
                         "atempting to create".format(outdir))
            if outdir is not "":
                os.makedirs(outdir)

    if args.scalesquared:
        args.scale = True

    plot_decay_constant(args)

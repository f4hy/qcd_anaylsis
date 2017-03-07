#!/usr/bin/env python2
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import logging
import argparse
import os
import numpy as np


from commonplotlib.auto_key import auto_key
from commonplotlib.plot_helpers import add_mc_lines
from commonplotlib.plot_helpers import add_vert_lines

from plotter1_0.add_chiral_fits import add_chiral_fit, add_boot_fit
from plotter2_0.add_model_fit import add_model_fit
from plotter2_0.plot_data import get_data, plot_data
from itertools import cycle, count

from data_params import scale

from plot_uk_data import add_uk_plot_data

from ensemble_data2_0.all_ensemble_data import ensemble_data, MissingData, NoStrangeInterp

plt.rc('text', usetex=True)

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


def count_pm():
    yield 0
    for i in count(1):
        yield i
        yield -i


offsets = count_pm()


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


def colors_and_legend(data_properties, legend_mode="betaLs", ylabel=None):

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
        if p.s_mass == "shifted":
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

    if legend_mode == "operator":
        color, mark, mfc = auto_key(len(ylabel))
        legend_label = r'{}'.format(ylabel)

    symbol = mpl.lines.Line2D([], [], color=color, mec=color, marker=mark, markersize=15,
                              linestyle="None", label=legend_label, mfc=mfc)
    # legend_handles.append(mpatches.Patch(color=beta_colors[p.beta], label=mylabel))
    handel = (color, mark, mfc)
    if handel not in added_handles:
        if legend_label is not None:
            added_handles.append(handel)
            legend_handles.append(symbol)

    return handel


def plot_ensemble_data(options):

    xmax = ymax = -10000
    ymin = 100000000

    fontsettings = dict(fontsize=60)

    fig, axe = plt.subplots(1)

    for es in options.ensembles:
        for yd in options.ydata:
            if options.offset:
                offset = next(offsets)*options.offset
            else:
                offset = 0
            try:
                ydata = get_data(es, yd, options)
                xdata = get_data(es, options.xdata, options)
            except NoStrangeInterp as interperror:
                logging.warn("for {} found error {}".format(yd, interperror))
                continue
            except MissingData:
                logging.warn("for {} data is missing data {} {}".format(es.ep, options.xdata, yd))
                continue

            if options.mhcut:
                ydata = {k: v for k, v in ydata.iteritems() if es.ep.heavies[k] < options.mhcut}

            if isinstance(xdata, plot_data) and isinstance(ydata, plot_data):
                plotdata = [(xdata, ydata)]
            elif isinstance(xdata, plot_data):
                logging.info("one x data but many y")
                plotdata = [(xdata, y) for m, y in ydata.iteritems()]
            elif isinstance(ydata, plot_data):
                plotdata = [(x, ydata) for m, x in xdata.iteritems()]
            else:
                logging.info("many of both data types")
                keys = set(xdata).intersection(ydata)
                plotdata = [(xdata[k], ydata[k]) for k in keys]
                if len(keys) < 1:
                    logging.error("Data types have no matching keys")
                    logging.error("xdata {}".format(xdata))
                    logging.error("ydata {}".format(ydata))
                    exit(-1)

            for x, y in plotdata:
                xlabel = x.label
                ylabel = y.label
                xphysical = x.physical
                yphysical = y.physical
                # p = y.dp
                label = ""
                # label = "$f_{}$ s{}".format(y.dp.flavor, y.dp.s_mass)

                color, mark, mfc = colors_and_legend(es.ep, options.legend_mode, ylabel)

                summary_lines.append("{}, {}, {}\n".format(es.ep, y.value, y.error))

                alpha = 1.0
                if "32x64" in repr(es.ep) and es.ep.ud_mass < 0.004:
                    alpha = 0.0
                    color = "#9999FF"
                    # continue

                plotsettings = dict(linestyle="none", c=color, marker=mark,
                                    label=label, ms=9, elinewidth=4,
                                    capsize=8, capthick=2, mec=color, mew=2,
                                    aa=True, mfc=mfc, fmt='o', ecolor=color,
                                    alpha=alpha)
                logging.info("plotting {} {} {}".format(x.value, y.value, y.error))

                if options.xerror is False:
                    xerr = None
                else:
                    xerr = x.error
                axe.errorbar(x.value + offset, y.value, yerr=y.error, xerr=xerr, zorder=0, **plotsettings)
                # axe.errorbar(x.value + offset, y.value, yerr=y.addscale_error(es.ep.scale_err), xerr=xerr, zorder=0, **plotsettings)
                ymax = max(ymax, y.value) #
                ymin = min(ymin, y.value)
                xmax = max(xmax, x.value)

    if options.physical:
        logging.info("plotting physical {} {}".format(xphysical, yphysical))
        matchingkeys = set(xphysical).intersection(set(yphysical))

        if len(matchingkeys) > 1:
            physiter = [(k, xphysical[k], yphysical[k]) for k in matchingkeys]
        else:
            physiter = [(k, xp, yphysical[k]) for k in yphysical for xl, xp in xphysical.iteritems()]
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
            for i in scale:
                physxplot = axe.axvline(1.0/scale[i], color=auto_key((i, None, None), check=False)[0],
                                        ls="--", lw=2, label=i)
        else:
            for i in scale:
                physxplot = axe.axvline(scale[i], color=auto_key((i, None, None), check=False)[0],
                                        ls="--", lw=2, label=i)

    if options.add_uk_data:
        logging.info("adding ukdata {}".format(options.add_uk_data))
        handles = add_uk_plot_data(axe, options.add_uk_data[0], options.add_uk_data[1])
        legend_handles.extend(handles)

    add_mc_lines(axe, options, auto_key)
    add_vert_lines(axe, options)

    for xline in options.xlines:
        axe.axvline(x=xline, linewidth=2, color='k', ls="dotted")
    for yline in options.ylines:
        axe.axhline(y=yline, linewidth=2, color='k', ls="dotted")

    if options.physx:
        physxplot = axe.axvline(xphysical, color='k', ls="--", lw=2, label="physical point")
        legend_handles.append(physxplot)

    if options.addpoint:
        logging.info("adding point {}".format(options.addpoint))
        px = float(options.addpoint[1])
        py = float(options.addpoint[2])
        physplot = axe.errorbar(px, py, yerr=0, marker="x",
                                ecolor="k", color="k", label=options.addpoint[0],
                                ms=15, elinewidth=3, capsize=8,
                                capthick=2, mec='k', mew=3, mfc='k',
                                zorder=100)
        symbol = mpl.lines.Line2D([], [], color="k", mec="k", marker="x", markersize=15, mew=3,
                                  linestyle="None", label=options.addpoint[0], mfc="k")
        legend_handles.append(symbol)

    for pt in options.adderrpoint:

        logging.info("adding point {}".format(pt))
        px = float(pt[1])
        py = float(pt[2])
        pyerr = float(pt[3])
        physplot = axe.errorbar(px, py, yerr=pyerr, marker="", # noqa
                                ecolor="k", color="k", label=pt[0],
                                ms=15, elinewidth=3, capsize=8, alpha=0.5,
                                capthick=2, mec='k', mew=3, mfc='k',
                                zorder=100)
        symbol = mpl.lines.Line2D([], [], color="k", mec="k", marker="x", markersize=15, mew=3,
                                  linestyle="None", label=pt[0], mfc="k")
        legend_handles.append(symbol)

    if options.addpoints:
        logging.info("adding points from {}".format(options.addpoints))
        with open(options.addpoints) as pointfile:
            for l in pointfile:
                x, y = l.split(",")
                axe.errorbar(float(x), float(y), yerr=0, color='k', markersize=15, ecolor='k', marker='s')
                logging.info("adding points {} {}".format(x, y))
                ymax = max(ymax, float(y))
                ymin = min(ymin, float(y))
                xmax = max(xmax, float(x))

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

    if options.legend_clear:
        del legend_handles[:]

    if options.chiral_fit_file:
        # del legend_handles[:]
        for i in options.chiral_fit_file:
            fit_lines = add_chiral_fit(axe, xran, i, options)
            if options.legendfits:
                legend_handles.extend(fit_lines)

    if options.model_fit_file:
        logging.info("adding model fit from {}".format(options.model_fit_file))
        for i in options.model_fit_file:
            fithandles = add_model_fit(axe, xran, i, options)
            if options.legendfits:
                legend_handles.extend(fithandles)

    if options.boot_fit_file:
        # del legend_handles[:]
        for i in options.boot_fit_file:
            chiral_line = add_boot_fit(axe, xran, i, options)
            if options.legendfits:
                legend_handles.extend(chiral_line)

    if options.xlabel:
        axe.set_xlabel(options.xlabel, labelpad=20, **fontsettings)
    else:
        axe.set_xlabel(xlabel, labelpad=20, **fontsettings)
        if "MeV^2" in xlabel:
            import matplotlib.ticker as ticker
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/(1000**2)))
            axe.xaxis.set_major_formatter(ticks)
            axe.set_xlabel(xlabel.replace("MeV", "GeV"), **fontsettings)
        if "1/MeV" in xlabel:
            import matplotlib.ticker as ticker
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(1000*x))
            axe.xaxis.set_major_formatter(ticks)
            axe.set_xlabel(xlabel.replace("1/MeV", "1/GeV"), **fontsettings)

    if options.ylabel:
        if options.scale:
            axe.set_ylabel("{} [MeV]".format(options.ylabel), labelpad=10, **fontsettings)
        else:
            axe.set_ylabel("{}".format(options.ylabel), labelpad=10, **fontsettings)
    else:
        axe.set_ylabel("{}".format(ylabel), labelpad=10, **fontsettings)
        if "MeV^(3/2)" in ylabel:
            import matplotlib.ticker as ticker
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:.6g}'.format(x/(1000**(3.0/2.0))))
            start, end = axe.get_ylim()
            first, second = axe.yaxis.get_ticklocs()[0:2]
            bot = np.floor(10*start/(1000**(3.0/2.0)))/10*(1000**(3.0/2.0))
            top = np.ceil(100*end/(1000**(3.0/2.0)))/100*(1000**(3.0/2.0))
            axe.yaxis.set_ticks(np.arange(bot, top, (0.1*1000**(3.0/2.0))))
            axe.yaxis.set_major_formatter(ticks)
            axe.set_ylabel(ylabel.replace("MeV^(3/2)", "GeV^(3/2)"), **fontsettings)
            plt.ylim(bot, top)

    axe.tick_params(axis='both', which='major', labelsize=35)

    if options.title:
        fig.suptitle(options.title.replace("_", " "), **fontsettings)

    def legsort(i):
        return i.get_label()

    if not options.nolegend:
        axe.legend(handles=sorted(legend_handles, key=legsort), loc=options.legendloc, handletextpad=0.1, handlelength=1,
                   fontsize=40, numpoints=1, fancybox=True, framealpha=0.5)
    if(options.output_stub):
        options.output_stub = options.output_stub.replace(".", "_")
        summaryfilename = options.output_stub + ".txt"
        logging.info("Writting summary to {}".format(summaryfilename))
        with open(summaryfilename, 'w') as summaryfile:
            for i in summary_lines:
                summaryfile.write(i)

        width = 17.0
        fig.set_size_inches(width, width*options.aspect)
        fig.tight_layout()
        file_extension = ".png"
        if options.eps:
            file_extension = ".eps"
        if options.pdf:
            file_extension = ".pdf"
        filename = options.output_stub + file_extension
        logging.info("Saving plot to {}".format(filename))
        ensure_dir(filename)
        plt.savefig(filename)
        if options.open_image:
            if file_extension == ".png":
                from subprocess import Popen
                Popen(["feh", filename])
        return

    logging.info("".join(summary_lines))
    plt.show()


def ensure_dir(filename):
    outdir = os.path.dirname(filename)
    if not os.path.exists(outdir):
        logging.info("directory for output {} does not exist, atempting to create".format(outdir))
        if outdir is not "":
            os.makedirs(outdir)


if __name__ == "__main__":
    example_usage = """Example Usage:  plot_ensemble_data.py SymDW* --ydata mpi --xdata fpi -o fpi_by_mpi -x 0 500 \n
    """
    parser = argparse.ArgumentParser(description="Read pickled ensemble fit data and plot it",
                                     epilog=example_usage)

    axis_choices = ["mud", "mud_s", "mpi", "mpisqr", "2mksqr-mpisqr", "mpisqr/mq", "xi", "mq"]
    legend_choices = ["betaLs", "betaL", "heavy", "smearing", "flavor", "strange", "betaheavy", "custom", "operator"]

    extention_group = parser.add_mutually_exclusive_group()
    extention_group.add_argument("--png", action="store_true", default=True,
                                 help="save as eps not png")
    extention_group.add_argument("--eps", action="store_true",
                                 help="save as eps not png")
    extention_group.add_argument("--pdf", action="store_true",
                                 help="save as pdf not png")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument('ensembles', metavar='es', type=str, nargs='+',
                        help='ensembles to plot')
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
    parser.add_argument("-uk", "--add_uk_data", required=False, nargs=2, metavar=("Xdata", "Ydata"),
                        help="Add ukdata points", default=None)
    parser.add_argument("--addpoint", required=False, nargs=3, metavar=("LABEL", "X", "Y"),
                        help="Add a point", default=None)
    parser.add_argument("--adderrpoint", required=False, action='append', nargs=4, metavar=("LABEL", "X", "Y", "err"),
                        help="Add a point", default=[])
    parser.add_argument("--addpoints", required=False, type=str,
                        help="Add points from file", default=None)
    parser.add_argument("--xaxis", required=False, choices=axis_choices,
                        help="what to set on the xaxis", default="mud")
    parser.add_argument("--legend_clear", action="store_true", required=False,
                        help="clear the legend of data points")
    parser.add_argument("--legendfits", action="store_true", required=False,
                        help="add fits to legend")
    parser.add_argument("--nolegend", action="store_true", required=False,
                        help="do not draw the legend")
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
    parser.add_argument("--xlines", required=False, type=float, action='append', default=[],
                        help="draw line at a given xvalue")
    parser.add_argument("--ylines", required=False, type=float, action='append', default=[],
                        help="draw line at a given yvalue")
    parser.add_argument("-I", "--interpolate", type=argparse.FileType('r'), required=False,
                        help="add interpolated lines")
    parser.add_argument("--chiral_fit_file", type=argparse.FileType('r'), required=False,
                        action='append', help="add chiral interpolated lines")
    parser.add_argument("--boot_fit_file", type=argparse.FileType('r'), required=False,
                        action='append', help="add bootstrap fit lines")
    parser.add_argument("--model_fit_file", type=argparse.FileType('r'), required=False,
                        action='append', help="add bootstrap fit lines")
    parser.add_argument("--model_fit_point", required=False, default=None, type=float,
                        help="add a point on the model fit")
    parser.add_argument("--model_finite_fits", type=str, required=False, default=[],
                        action='append', help="add fit lines for finite beta")
    parser.add_argument("--mpisqrbymq", action="store_true",
                        help="compute mpisqr divided by mq, strange edge case")
    parser.add_argument("--ydata", required=False, type=str, action='append',
                        help="which data to plot on the yaxis", default=[])
    parser.add_argument("--xdata", required=False, type=str,
                        help="what to use as the xaxis", default="mud")
    parser.add_argument("--aspect", type=float, default=1.0, required=False,
                        help="determine the plot aspect ratio")
    parser.add_argument("--interpstrange", action="store_true",
                        help="use interpoalted strange masses", default=None)
    parser.add_argument("--mhcut", type=float, default=None, required=False,
                        help="cut of mh to fade")
    parser.add_argument("--open_image", action="store_true",
                        help="open the created image")
    parser.add_argument("--fittype", required=False, type=str,
                        help="what fittype files to read", default="uncorrelated")
    parser.add_argument("--offset", required=False, type=float, default=0.0,
                        help="offset differnt plots from each other")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    ensembles = []
    for es in args.ensembles:
        ed = ensemble_data(es, fittype=args.fittype)
        ensembles.append(ed)

    logging.info("Ploting data for: {}".format("\n".join(args.ensembles)))
    args.ensembles = ensembles

    if args.output_stub:
        outdir = os.path.dirname(args.output_stub)
        if not os.path.exists(outdir):
            logging.info("directory for output {} does not exist,"
                         "atempting to create".format(outdir))
            if outdir is not "":
                os.makedirs(outdir)

    # if args.scalesquared:
    #     args.scale = True
    plot_ensemble_data(args)

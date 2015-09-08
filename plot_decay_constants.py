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
from ensamble_info import phys_pion, phys_kaon, phys_mq
from ensamble_info import Zs, Zv

from auto_key import auto_key


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

    if xaxis_type == "mpisqr/mq":
        pionmass = read_fit_mass(data_properties, "ud-ud", options.fitdata)
        mq = data_properties.ud_mass * 1.0/Zs[data_properties.beta]
        return (s*pionmass)**2 / (s*mq)

    if xaxis_type == "mq":
        residual = residual_mass(data_properties.ud_mass, data_properties.s_mass)
        mq = (data_properties.ud_mass+residual) * 1.0/Zs[data_properties.beta]
        return np.array((s*mq))



legend_handles = []
added_handles = []
summary_lines = []
s_mass_marks = {}
markers = ['o', "D",  "<",  "p", "8", "v", "^", "D", ">", 'o']
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


def colors_and_legend(data_properties, legend_mode="betaLs"):

    p = data_properties
    if legend_mode == "betaLs":
        color, mark, mfc = auto_key((p.beta, p.s_mass, p.latsize))
        legend_label = r'$\beta = {}, L={}, m_s={}$'.format(p.beta, p.latsize[:2], p.s_mass)

    if legend_mode == "betaL":
        color, mark, mfc = auto_key((p.beta, p.s_mass))
        legend_label = r'$\beta = {}, L={}$'.format(p.beta, p.latsize[:2])


    if legend_mode == "strange":
        color, mark, mfc = auto_key(p.s_mass)
        legend_label = r'$ms={}$'.format(p.s_mass)

    if legend_mode == "heavy":
        color, mark, mfc = auto_key(p.heavymass)
        legend_label = r'$mh={}$'.format(p.heavymass)

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
        added_handles.append(handel)
        legend_handles.append(symbol)

    return handel


def add_interpolate(axe, xran, fit_file, chiral_fit_file=None):

    values = {}
    errors = {}

    phys_mpisqr = (phys_pion)**2

    for i in fit_file:
        if i.startswith("#"):
            chisqr_dof = float(i.split(" ")[-1])
            continue

        name, val, err = (j.strip() for j in i.replace("+/-",",").split(","))

        values[name] = float(val)
        errors[name] = float(err)

    logging.debug(values)
    logging.debug(errors)

    x =  np.linspace(phys_mpisqr, xran[1])


    #y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)*np.log(1.0/x)
    #y = values["F_PI"]+values["M_pi"]*(x-phys_mpisqr)*np.log(1.0/x)
    y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)

    yerr = errors["phys_obs"]+errors["M_pi"]*(x-phys_mpisqr)

    print x
    print y
    print yerr
    p = axe.plot(x,y, label="Linear fit $\chi^2$/dof:{:.2}".format(chisqr_dof), color='g', lw=2)
    # hbar_c = 197.327
    # y_417 = y + +values["A"]*((hbar_c/scale["4.17"])**2)
    # color = auto_key(("4.17", 0, 0))[0]
    # p.extend(axe.plot(x,y_417, color=color, ls="-.", label="fit at a^2 for beta=4.17"))

    # axe.fill_between(x,y, y+yerr, facecolor='g', alpha=0.1, lw=0, zorder=-10)
    # axe.fill_between(x,y, y-yerr, facecolor='g', alpha=0.1, lw=0, zorder=-10)

    axe.errorbar(phys_mpisqr, values["phys_obs"], yerr=errors["phys_obs"], color='g', elinewidth=4, capsize=8,
                 capthick=2, mec='g', mew=2)

    return p

def add_chiral_fit(axe, xran, chiral_fit_file=None):

    values = {}
    errors = {}

    for i in chiral_fit_file:
        if i.startswith("#"):
            continue

        name, val, err = (j.strip() for j in i.replace("+/-",",").split(","))
        values[name] = float(val)
        errors[name] = float(err)


    if "OMEGA_F" in values.keys():
        return add_NLO_chiral_fit(axe, xran, values, errors)
    else:
        return add_LO_chiral_fit(axe, xran, values, errors)

def add_LO_chiral_fit(axe, xran, values, errors):

    phys_mpisqr = (phys_pion)**2


    rho_mass = 775.4
    LAMBDA = values["LAMBDA"]

    x =  np.linspace(phys_mpisqr, xran[1])

    XI = x/(8*(np.pi**2)*(values["F_PI"])**2)
    y = values["F_PI"] * (1 - XI*np.log(x/(LAMBDA)**2))
    # y = values["F_PI"] * (1 - XI*np.log(x/(LAMBDA)**2)
    #                       - 1.0/4.0 * (XI**2)*(np.log(x/values["OMEGA_F"]**2))**2  )

    # y = values["F_PI"] * (1 - XI*np.log(x/(LAMBDA))
    #                       - 1.0/4.0 * (XI**2)*(np.log(x/values["OMEGA_F"]))**2 + values["c_F"]*XI**2 )


    #yerr = errors["F_PI"] + errors["F_PI"]*x/(8*np.pi*values["F_PI"])**2*np.log(x/(LAMBDA)**2)

    p = axe.plot(x,y, label="LO chiral fit", color='b', ls="--", lw=2)
    #plt.show()
    return p

    #
    # y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)+values["A"]*((hbar_c/scale["4.17"])**2)
    # axe.plot(x,y, ls="-")
    # y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)+values["A"]*((hbar_c/scale["4.35"])**2)
    # axe.plot(x,y, ls="--")
    # y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)+values["A"]*((hbar_c/scale["4.47"])**2)
    # axe.plot(x,y, ls="-.")

    # # print auto_key("4.17")
    # # print auto_key("4.35")
    # foobar = auto_key("4.47")
    # print "fobarbaz"
    # print "4.47", foobar
    # print foobar, "test", "4.47"
    # exit(-1)

def add_NLO_chiral_fit(axe, xran, values, errors):


    phys_mpisqr = (phys_pion)**2


    rho_mass = 775.4
    LAMBDA = values["LAMBDA"]

    x =  np.linspace(phys_mpisqr, xran[1])

    XI = x/(8*(np.pi**2)*(values["F_PI"])**2)
    y = values["F_PI"] * (1 - XI*np.log(x/(LAMBDA)**2)
                          - 1.0/4.0 * (XI**2)*(np.log(x/values["OMEGA_F"]**2))**2  )

    # y = values["F_PI"] * (1 - XI*np.log(x/(LAMBDA))
    #                       - 1.0/4.0 * (XI**2)*(np.log(x/values["OMEGA_F"]))**2 + values["c_F"]*XI**2 )


    #yerr = errors["F_PI"] + errors["F_PI"]*x/(8*np.pi*values["F_PI"])**2*np.log(x/(LAMBDA)**2)

    p = axe.plot(x,y, label="NLO chiral fit", color='m', ls="-.", lw=2)
    #plt.show()
    return p

    #
    # y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)+values["A"]*((hbar_c/scale["4.17"])**2)
    # axe.plot(x,y, ls="-")
    # y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)+values["A"]*((hbar_c/scale["4.35"])**2)
    # axe.plot(x,y, ls="--")
    # y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)+values["A"]*((hbar_c/scale["4.47"])**2)
    # axe.plot(x,y, ls="-.")

    # # print auto_key("4.17")
    # # print auto_key("4.35")
    # foobar = auto_key("4.47")
    # print "fobarbaz"
    # print "4.47", foobar
    # print foobar, "test", "4.47"
    # exit(-1)


def plot_decay_constant(options):


    #plt.rc('text', usetex=True)

    xmax = ymax = -10000
    ymin = 100000000

    fontsettings = dict(fontsize=30)

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

        if "decay" in f:
            logging.info("Is a decay constant file")
            names = ["decay"]
            dataname = "decay"
        else:
            names = ["bootstrap", "mass", "amp1", "amp2"]
            dataname = "mass"

        df = pd.read_csv(f,comment='#', names=names)

        if options.mpisqrbymq:
            if options.scale:
                res = residual_mass(p.ud_mass, p.s_mass)
                df[dataname] = ((scale[p.beta]*df[dataname])**2 / (scale[p.beta]*(p.ud_mass+res) * 1.0/Zs[p.beta]))
            else:
                df[dataname] = ((df[dataname])**2 / (p.ud_mass * 1.0/Zs[p.beta]))


        mark = markers[index % len(markers)]

        mfc = 'white'

        mark = strange_legend(p.s_mass)

        color, mark, mfc = colors_and_legend(p, options.legend_mode)

        # if "48x96x12" in f:
        #     logging.info("48x96x12!!!!")
        #     color = 'g'
        #     latsize = re.search("_([0-9]*x[0-9]*x[0-9]*)_", f).group(1)
            # if latsize not in added_handles:
            #     legend_handles.append(mpatches.Patch(color=color, label=r"$\beta = 4.17 48x96$"))
            #     added_handles.append(latsize)

        alpha = 1.0
        if "32x64" in f and p.ud_mass < 0.004:
            alpha = 0.6
            color="#9999FF"

        xs = xvalues(options.xaxis, p, options)
        x = xs.mean()
        xerr = xs.std()
        xerr = plot_helpers.error(xs)
        y = float(df[dataname].mean())
        e = float(df[dataname].std())
        e = plot_helpers.error(df[dataname])

        summary_lines.append("{}, {}, {}\n".format(p, y, df[dataname].std()))

        scalepower = 1.0
        if options.scalesquared:
            scalepower = 2.0

        # if has_shifted and p.s_mass != "interpolated":
        #     alpha = 0.3

        if p.ratio:
            scalepower = 0.0

        plotsettings = dict(linestyle="none", c=color, marker=mark, label=label, ms=12, elinewidth=4, capsize=8,
                            capthick=2, mec=color, mew=2, aa=True, mfc=mfc, fmt='o', ecolor=color, alpha=alpha)
        index+=1
        logging.info("plotting {} {} {}".format(x,y,e))
        if options.scale:
            sc = scale[p.beta]**scalepower
            if options.mpisqrbymq:
                sc=1.0
            scaled_err = plot_helpers.error(df[dataname]*sc)

            if options.box:
                b = axe.boxplot(df[dataname]*sc, positions=[x], widths=[0.001*sc], patch_artist=True)
            elif options.scatter:
                axe.scatter(xs, df[dataname]*sc, c=color)
            else:
                axe.errorbar(x, y*sc, yerr=scaled_err, zorder=0, **plotsettings)
            ymax = max(ymax,y*sc)
            ymin = min(ymin,y*sc)
            xmax = max(xmax,x)

        else:
            if options.box:
                b = axe.boxplot(df[dataname], positions=[x], widths=[0.001], patch_artist=True)
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
        x_physicals = {"mud": 2.2, "mud_s": 97.2, "mpisqr": phys_pion**2, "2mksqr-mpisqr": 2*(phys_kaon**2)-phys_pion**2, "mpisqr/mq": (phys_pion**2)/(phys_mq), "mq": (phys_mq)}
        y, err = options.physical
        if options.scale:
            x = x_physicals[options.xaxis]
        else:
            x = 0.001
        physplot = axe.errorbar(x, y, yerr=err, marker="x",
                                ecolor="k", color="k", label="PDG",
                                ms=15, elinewidth=3, capsize=1,
                                capthick=2, mec='k', mew=3, mfc='k',
                                zorder=100)
        symbol = mpl.lines.Line2D([], [], color="k", mec="k", marker="x", markersize=15, mew=3,
                                  linestyle="None", label="PDG", mfc="k")
        legend_handles.append(symbol)
        ymax = max(ymax,y)
        ymin = min(ymin,y)


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
            plt.ylim(auto_fit_range(ymin, ymax))


    # if options.title:
    #     axe.set_title(options.title, **fontsettings)

    #axe.set_xlabel("$m_{%s}+m_{res}+m_{%s}+m_{res}$" % (rawflavor.split("-")[0], rawflavor.split("-")[1]), **fontsettings)

    xlabel = {"mud": u"$m_{l}+m_{res}$", "mud_s": u"$m_{l}+m_s+2m_{res}$", "mpi": u"$m_{\pi}$",
              "mpisqr": u"$m^2_{\pi}$", "2mksqr-mpisqr": u"$2m^2_{K}-m^2_{\pi}$", "mpisqr/mq": u"$m^2_{\pi}/m_q$", "mq": u"$m_q$" }

    if options.scale:
        xlabel = {"mud": u"$m_{l}+m_{res}$", "mud_s": u"$m_{l}+m_s+2m_{res}$", "mpi": u"$m_{\pi}$",
                  "mpisqr": u"$m^2_{\pi}$ [MeV^2]", "2mksqr-mpisqr": u"$2m^2_{K}-m^2_{\pi}$", "mpisqr/mq": u"$m^2_{\pi}/m_q$ [MeV]", "mq": u"$m_q$ [MeV]" }


    if options.xaxis == "mpisqr":
        if options.interpolate:
            interp_line = add_interpolate(axe, xran, options.interpolate)
            legend_handles.extend(interp_line)

        if options.chiral_fit_file:
            for i in options.chiral_fit_file:
                chiral_line = add_chiral_fit(axe, xran, i)
                legend_handles.extend(chiral_line)


    if options.xlabel:
        axe.set_xlabel(options.xlabel, **fontsettings)
    else:
        axe.set_xlabel(xlabel[options.xaxis], **fontsettings)


    if options.ylabel:
        if options.scale:
            axe.set_ylabel("{} [MeV]".format(options.ylabel), **fontsettings)
        else:
            axe.set_ylabel("{}".format(options.ylabel), **fontsettings)
    elif options.scale:
        if options.scalesquared:
            axe.set_ylabel("MeV^2", **fontsettings)
        else:
            axe.set_ylabel("MeV", **fontsettings)
    else:
        axe.set_ylabel("lattice units", **fontsettings)

    axe.tick_params(axis='both', which='major', labelsize=20)

    def legsort(i):
        return i.get_label()

    if not options.box:
        leg = axe.legend(handles=sorted(legend_handles, key=legsort), loc=0, fontsize=20, numpoints=1)
    if(options.output_stub):
        summaryfilename = options.output_stub + ".txt"
        logging.info("Writting summary to {}".format(summaryfilename))
        with open(summaryfilename, 'w') as summaryfile:
            for i in summary_lines:
                summaryfile.write(i)

        fig.set_size_inches(18.5, 10.5)
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

    axis_choices = ["mud", "mud_s", "mpi", "mpisqr", "2mksqr-mpisqr", "mpisqr/mq", "xi", "mq" ]
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
    parser.add_argument("--chiral_fit_file", type=argparse.FileType('r'), required=False, action='append',
                        help="add chiral interpolated lines")
    parser.add_argument("--mpisqrbymq", action="store_true",
                        help="compute mpisqr divided by mq, strange edge case")
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

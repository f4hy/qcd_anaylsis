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

from residualmasses import residual_masses

from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from ensamble_info import all_same_beta, all_same_heavy, all_same_flavor
from ensamble_info import phys_pion, phys_kaon





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
        residual = residual_masses[(data_properties.ud_mass, data_properties.s_mass)]
        return pd.Series(s*(data_properties.ud_mass + residual))

    if xaxis_type == "mud_s":
        residual = residual_masses[(data_properties.ud_mass, data_properties.s_mass)]
        return mp.Series(s*(data_properties.ud_mass + residual + data_properties.s_mass + residual))

    if xaxis_type == "mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", options.fitdata)
        return (s*pionmass)**2

    if xaxis_type == "2mksqr-mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", options.fitdata)
        kaonmass = read_fit_mass(data_properties, "ud-s", options.fitdata)
        return 2.0*(s*kaonmass)**2 - (s*pionmass)**2




legend_handles = []
added_handles = []
summary_lines = []
s_mass_marks = {}
markers = ['o', "D", "^", "<", ">", "v", "x", "p", "8", 'o', "D"]
colors = ['b', 'r', 'k', 'm', 'c', 'y', 'b', 'r', 'k', 'm', 'c', 'y']*2

beta_colors = {"4.17": 'b', "4.35": 'r', "4.47": 'm'}

heavy_colors = {"m0": 'b', "m1": 'r', 'm2': 'm', 's0': 'c'}
heavy_colors = {}
s_mass_colors = {}
smearing_colors = {}


flavor_color = {"\pi": 'b', "K": 'r', '\eta': 'm', "Hl": 'c', "Hs": 'y'}


def colors_and_legend(data_properties, legend_mode="beta"):


    p = data_properties

    if p.s_mass not in s_mass_marks.keys():
        s_mass_marks[p.s_mass] = markers.pop()
        mark = s_mass_marks[p.s_mass]
        smass_leg = mlines.Line2D([], [], color='black', marker=s_mass_marks[p.s_mass], mfc='white', mew=3, lw=0,
                                markersize=18, label='$m_s={}$'.format(p.s_mass))
        #legend_handles.append(smass_leg)
        #added_handles.append(p.s_mass)
    else:
        mark = s_mass_marks[p.s_mass]


    if legend_mode == "strange":
        if p.s_mass not in added_handles:
            s_mass_colors[p.s_mass] = colors.pop()
            color = s_mass_colors[p.s_mass]
            legend_handles.append(mpatches.Patch(color=color, label='$m_s:{}$'.format(p.s_mass)))
            added_handles.append(p.s_mass)
        else:
            color = s_mass_colors[p.s_mass]
        return mark, color


    if legend_mode == "heavy":
        if p.heavymass not in added_handles:
            heavy_colors[p.heavymass] = colors.pop()
            color = heavy_colors[p.heavymass]
            legend_handles.append(mpatches.Patch(color=color, label='${}$'.format(p.heavymass)))
            added_handles.append(p.heavymass)
        else:
            color = heavy_colors[p.heavymass]
        return mark, color


    if legend_mode == "smearing":
        logging.info("Only one beta nd one flavor given, using smearing")
        if p.smearing not in added_handles:
            smearing_colors[p.smearing] = colors.pop()
            color = smearing_colors[p.smearing]
            legend_handles.append(mpatches.Patch(color=color, label='${}$'.format(p.smearing)))
            added_handles.append(p.smearing)
        else:
            color = smearing_colors[p.smearing]
        return mark, color


    if legend_mode == "flavor":
        color = flavor_color[p.flavor]
        if p.flavor not in added_handles:
            legend_handles.append(mpatches.Patch(color=color, label='${}$'.format(p.flavor)))
            added_handles.append(p.flavor)
        return mark, color

    if legend_mode == "beta":
        color = beta_colors[p.beta]
        if p.beta not in added_handles:
            mylabel = r'$\beta = {}$'.format(p.beta)
            legend_handles.append(mpatches.Patch(color=beta_colors[p.beta], label=mylabel))
            added_handles.append(p.beta)
        return mark, color


def read_bootstraps(f, options):


    df = pd.read_csv(f,comment='#', names=["config", "mass", "amp1", "amp2"])

    if options.spinaverage:
        if "vectorave" not in f:
            raise SystemExit("spin average requires inputing the vector file")
        ppfile = f.replace("vectorave", "PP")
        PPdf = pd.read_csv(ppfile,comment='#', names=["config", "mass", "amp1", "amp2"])
        return (3.0*df["mass"] + PPdf["mass"])/4.0
    else:
        return df["mass"]


def add_interpolate(axe, xran, fit_file):

    values = {}
    errors = {}

    phys_mpisqr = (138.04)**2

    for i in fit_file:
        if i.startswith("#"):
            chisqr_dof = float(i.split(" ")[-1])
            continue

        name, val, err = (j.strip() for j in i.replace("+/-",",").split(","))

        values[name] = float(val)
        errors[name] = float(err)

    print values
    print errors

    x =  np.linspace(phys_mpisqr, xran[1])

    y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)

    print x
    print y

    p = axe.plot(x,y, label="linear fit at a^2=0 $\chi^2$/dof:{:.2}".format(chisqr_dof), color='g', lw=2)
    hbar_c = 197.327
    y = values["phys_obs"]+values["M_pi"]*(x-phys_mpisqr)+values["A"]*((hbar_c/scale["4.17"])**2)
    color = "b"
    p.extend(axe.plot(x,y, color=color, ls="-.", label="linear fit at a^2 for beta=4.17", lw=2))


    axe.errorbar(phys_mpisqr, values["phys_obs"], yerr=errors["phys_obs"], color='g', elinewidth=4, capsize=8,
                 capthick=2, mec='g', mew=2)

    return p


def plot_mass(options):


    #plt.rc('text', usetex=True)

    xmax = ymax = -10000
    ymin = 10000

    fontsettings = dict(fontsize=30)

    flavor_patches = [mpatches.Patch(color=c, label='${}$'.format(l)) for l,c in flavor_color.iteritems() ]
    heavy_patches = [mpatches.Patch(color=c, label='${}$'.format(l)) for l,c in flavor_color.iteritems() ]

    has_shifted = any("msshifted" in f for f in options.files)

    one_heavy = all_same_heavy(options.files)
    one_beta = all_same_beta(options.files)
    one_flavor = all_same_flavor(options.files)
    logging.info("one_beta: {}, one_flavor: {}".format(one_beta, one_flavor))

    data = {}
    index = 0
    fig, axe = plt.subplots(1)


    for f in options.files:

        p = data_params(f)

        label = "$f_{}$ s{}".format(p.flavor, p.s_mass)

        data = read_bootstraps(f, options)



        mark = markers[index % len(markers)]
        # color = colors[index % len(colors)]

        mfc = 'white'

        # mark = strange_legend(p.s_mass)

        mark, color = colors_and_legend(p, options.legend_mode)

        # if "48x96x12" in f:
        #     logging.info("48x96x12!!!!")
        #     color = 'g'
        #     latsize = re.search("_([0-9]*x[0-9]*x[0-9]*)_", f).group(1)
            # if latsize not in added_handles:
            #     legend_handles.append(mpatches.Patch(color=color, label=))
            #     added_handles.append(latsize)


        xs = xvalues(options.xaxis, p, options)
        x = xs.median()
        xerr = xs.std()

        if "32x64" in f and p.ud_mass < 0.004:
            alpha = 0.6
            color="#9999FF"

        alpha = 1.0
        if has_shifted and p.s_mass != "shifted":
            alpha = 0.3

        plotsettings = dict(linestyle="none", c=color, marker=mark, label=label, ms=8, elinewidth=3, capsize=8,
                            capthick=2, mec=color, mew=3, aa=True, mfc=mfc, fmt='o', ecolor=color, alpha=alpha)
        index+=1

        y = data.mean()
        e = data.std()

        summary_lines.append("{}, {}, {}\n".format(p, y, e))

        logging.info("plotting {} {} {}".format(x,y,e))
        if options.scale:
            if options.box:
                b = axe.boxplot(data*scale[p.beta], positions=[x], widths=[0.001*scale[p.beta]], patch_artist=True)
            else:
                axe.errorbar(x, y*scale[p.beta], yerr=e*scale[p.beta], zorder=0, **plotsettings)
            ymax = max(ymax,y*scale[p.beta])
            ymin = min(ymin,y*scale[p.beta])
            xmax = max(xmax,x)
        else:
            if options.box:
                b = axe.boxplot(data, positions=[x], widths=[0.001], patch_artist=True)
            else:
                axe.errorbar(x, y, yerr=e, zorder=0, **plotsettings)
            ymax = max(ymax,y)
            ymin = min(ymin,y)
            xmax = max(xmax,x)

        if options.box:
            b["boxes"][0].set_color(color)
            b["boxes"][0].set_facecolor(color)
            b["medians"][0].set_color('k')
            plt.setp(b["fliers"], visible=False)


    if options.physical:
        x_physicals = {"mud": 2.2, "mud_s": 97.2, "mpisqr": phys_pion**2, "2mksqr-mpisqr": 2*(phys_kaon**2)-phys_pion**2}
        y, err = options.physical
        physplot = axe.errorbar(x_physicals[options.xaxis], y, yerr=err, marker="x", ecolor="k",
                                color="k", label="PDG", ms=15, elinewidth=3, capsize=1, capthick=2,
                                mec="k", mew=3, mfc='k', zorder=100)
        symbol = mpl.lines.Line2D([], [], color="k", mec="k", marker="x", markersize=15,
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
            plt.ylim(auto_fit_range(ymin, ymax, buff=3.5))
        else:
            plt.ylim(auto_fit_range(ymin, ymax))

    if options.title:
        axe.set_title(options.title, **fontsettings)

    if options.interpolate:
        interp_line = add_interpolate(axe, xran, options.interpolate)
        legend_handles.extend(interp_line)


    #axe.set_xlabel("$m_{%s}+m_{res}+m_{%s}+m_{res}$" % (rawflavor.split("-")[0], rawflavor.split("-")[1]), **fontsettings)

    xlabel = {"mud": u"$m_{l}+m_{res}$", "mud_s": u"$m_{l}+m_s+2m_{res}$", "mpi": u"$m_{\pi}$",
              "mpisqr": u"$m^2_{\pi}$ [MeV^2]", "2mksqr-mpisqr": u"$2m^2_{K}-m^2_{\pi}$" }

    axe.set_xlabel(xlabel[options.xaxis], **fontsettings)

    if options.ylabel:
        axe.set_ylabel("{} [MeV]".format(options.ylabel), **fontsettings)
    elif options.scale:
        axe.set_ylabel("MeV", **fontsettings)
    else:
        axe.set_ylabel("lattice units", **fontsettings)

    axe.tick_params(axis='both', which='major', labelsize=20)

    def legsort(i):
        return i.get_label()

    if not options.box:
        leg = axe.legend(handles=sorted(legend_handles, key=legsort), loc=0, fontsize=20, numpoints=1 )
    if(options.output_stub):
        summaryfilename = options.output_stub + ".txt"
        logging.info("Writting summary to {}".format(summaryfilename))
        with open(summaryfilename, 'w') as summaryfile:
            for i in summary_lines:
                summaryfile.write(i)
        fig.set_size_inches(18.5, 10.5)
        file_extention = ".png"

        if args.eps:
            file_extention = ".eps"
        if args.pdf:
            file_extention = ".pdf"
        filename = options.output_stub+file_extention
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename)
        return

    print "".join(summary_lines)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="average data files")

    axis_choices = ["mud", "mud_s", "mpi", "mpisqr", "2mksqr-mpisqr"]
    legend_choices = ["beta", "strange", "flavor", "heavy"]

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
    parser.add_argument("-y", "--yrange", type=float, required=False, nargs=2,
                        help="set the yrange of the plot", default=None)
    parser.add_argument("-x", "--xrange", type=float, required=False, nargs=2,
                        help="set the xrange of the plot", default=None)
    parser.add_argument("--xaxis", required=False, choices=axis_choices,
                        help="what to set on the xaxis", default="mud")
    parser.add_argument("--legend_mode", required=False, choices=legend_choices,
                        help="what to use for the legend", default="beta")
    parser.add_argument("--ylabel", type=str, required=False,
                        help="ylabel", default=None)
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument("-t", "--title", type=str, required=False,
                        help="plot title", default="masses")
    parser.add_argument("-s", "--scale", action="store_true",
                        help="scale the values")
    parser.add_argument("--spinaverage", action="store_true",
                        help="spinaverage vector with pseudoscalar")
    parser.add_argument("-p", "--physical", type=float, nargs=2,
                        help="add physical point")
    parser.add_argument("-I", "--interpolate", type=argparse.FileType('r'), required=False,
                        help="add interpolated lines")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info("Ploting mass for: {}".format("\n".join(args.files)))

    if args.output_stub:
        outdir = os.path.dirname(args.output_stub)
        if not os.path.exists(outdir):
            logging.info("directory for output {} does not exist, atempting to create".format(outdir))
            if outdir is not "":
                os.makedirs(outdir)


    plot_mass(args)

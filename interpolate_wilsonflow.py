#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import leastsq
import glob

colors = ['b', 'k', 'c', 'y', 'm', 'b']
#colors = ['c', 'm', 'k', 'm', 'c', 'y', 'b', 'r', 'k', 'm', 'c', 'y']

from ensamble_info import flavor_map, scale, data_params, determine_flavor, read_fit_mass
from ensamble_info import all_same_beta, all_same_heavy, all_same_flavor


def read_files(files):
    data = {}

    N = None

    for f in files:
        print f
        mud = float(re.search("_mud(0\.[0-9]*)_", f).group(1))
        strange_mass = re.search("ms([a-z0-9.]+)", f).group(1)
        strange_mass = strange_mass.replace(".jack", "")
        try:
            strange_mass = float(strange_mass)
        except:
            pass

        if strange_mass == "shifted":
            strange_mass = "interpolated"

        if strange_mass not in data.keys():
            data[strange_mass] = []

        df = pd.read_csv(f, comment='#', names=["flow"])

        if N is None:
            N = len(df.flow.values)
        else:
            if len(df.flow.values) != N:
                logging.warn("Size missmatch in {}!".format(f))
                continue

        data[strange_mass].append((mud, df.flow.values))


    return data


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



def get_physical_point(physical, line_params):
    m, b = line_params
    mcc = (physical - b)/m
    return mcc


def interpolate(data, funct="line", physical=None):

    logging.info("using the physical value {}".format(physical))

    muds = []
    flows = []
    fits = []


    for mud, flow in data:
        muds.append(mud)
        flows.append(flow)


    Nconfigs = len(flows[0])

    def line(v, x, y):
        return (v[0]*x+v[1]) - y

    def quad(v, x, y):
        return (v[0]*x+v[1]+v[2]*(x*x)) - y

    logging.info("Using {} jackknife samples".format(Nconfigs))
    for i in range(Nconfigs):
        A = np.array(muds)

        fl = [f[i] for f in flows]
        B = np.array(fl)

        slope_guess = (min(B)-max(B)) / (max(A) - min(A))
        int_guess = min(B) - min(A)*slope_guess
        line_guess = [slope_guess, int_guess]
        quad_guess = [slope_guess, int_guess, 0.0]

        if funct == "line":
            logging.debug("guessing a line with y={}x+{}".format(*line_guess))
            best_fit, _, info, mesg, ierr = leastsq(line, line_guess, args=(A, B), maxfev=10000, full_output=True)
        elif funct == "quad":
            logging.debug("guessing a line with y={2}x^2+{0}x+{1}".format(*quad_guess))
            best_fit, _, info, mesg, ierr = leastsq(quad, quad_guess, args=(A, B), maxfev=10000, full_output=True)

        fits.append(best_fit)

    return fits


def gen_line_func(parameters):

    if len(parameters) == 2:
        def fitline(x):
            return parameters[0] * x + parameters[1]
    else:
        def fitline(x):
            return parameters[0] * x + parameters[1] + parameters[2] * (x*x)


    return fitline


def plot_fitline(data, fitlines, med_line, intersects, a_invs, ftype, beta, strangeness, outstub):
    size = 100

    c = colors.pop()
    fontsettings = dict(fontsize=20)
    plotsettings = dict(linestyle="none", ms=8, elinewidth=3, capsize=8,
                        capthick=2, mew=3, aa=True, fmt='o')

    muds = []
    for mud, flow in data:
        N = len(flow)
        y = np.mean(flow)
        err = np.sqrt((N-1)*(np.std(flow)**2))
        logging.info("mud={}, {}={}, err={}".format(mud, ftype, y, err))
        patch = plt.errorbar(mud, y, yerr=err, color=c, ecolor=c, mec=c, **plotsettings)



        muds.append(mud)
        #plt.scatter(heavymass, mesonmass, s=size)

    xdata = np.arange(-0.001, max(muds)+0.005, 0.001)
    mydata = np.array([med_line(x) for x in xdata])
    plt.plot(xdata, mydata, color=c)

    it_y = np.mean(intersects)
    it_err = np.sqrt((N-1)*(np.std(intersects)**2))
    plt.errorbar(0, it_y, it_err, color="r", mec="r", **plotsettings)


    filldatas = []
    erryp = []
    errym = []
    for x in xdata:
        y = med_line(x)
        fillys = [y-f(x) for f  in fitlines]
        erryp.append(y - np.sqrt((N-1)*np.std(fillys)**2))
        errym.append(y + np.sqrt((N-1)*np.std(fillys)**2))

    plt.fill_between(xdata, mydata, erryp, facecolor=c, alpha=0.3, lw=0, zorder=-10)
    plt.fill_between(xdata, mydata, errym, facecolor=c, alpha=0.3, lw=0, zorder=-10)


    # logging.info("physical point {}, {}".format(0, intersects))

    #plt.scatter(0, at_zero, c='r', s=size)

    if a_invs[0] is not None:
        a_inv = np.mean(a_invs)
        err_a_inv = np.sqrt((N-1)*(np.std(a_invs)**2))

        text = "{}: {}  -> 1/a = {:.2f} +/- {:.2f} MeV".format(strangeness, it_y, a_inv, err_a_inv)
        logging.info(text)

        #plt.annotate(text, xy=(0,it_y), xytext=(0.5, it_y), textcoords="axes fraction", arrowprops=dict(facecolor="black"), **fontsettings)
        return mpatches.Patch(color=c, label=text)
    return mpatches.Patch(color=c, label="{}".format(strangeness))


def finish_plot(beta, ftype, legend_handles, outstub):
    fontsettings = dict(fontsize=20)

    plotsettings = dict(linestyle="none", ms=8, elinewidth=3, capsize=8,
                    capthick=2, mew=3, aa=True, fmt='o')

    plt.title(r'$\beta={}$    ${}$'.format(beta, ftype), **fontsettings)
    plt.ylabel("${} / a$".format(ftype), **fontsettings)
    plt.xlabel("m_ud", **fontsettings)

    plt.legend(handles=sorted(legend_handles), loc=0, **fontsettings )

    if outstub is not None:
        fig = plt.gcf()
        fig.set_size_inches(18.5,10.5)
        filename = outstub+".png"
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


def write_data(intersects, a_invs, output_stub, suffix):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing a_inv to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        try:
            ofile.write("# mean: {}, {}\n".format(np.mean(intersects), np.mean(a_invs)))
        except:
            ofile.write("# mean: {}\n".format(np.mean(intersects)))

        for i,a in zip(intersects,a_invs):
            ofile.write("{}, {}\n".format(i, a))



def interpolate_wilsonflow(options):
    """ script to interpolate the heavy mass """
    logging.debug("Called with {}".format(options))

    try:
        beta = re.search("_b(4\.[0-9]*)_", options.files[0]).group(1)
    except:
        logging.error("failed to determine beta for {}".format(options.files[0]))
        exit(-1)

    if "t0" in options.files[0]:
        ftype = "t_0^{1/2}"
    elif "w0" in options.files[0]:
        ftype = "w_0"
    else:
        ftype = None

    alldata = read_files(options.files)

    legend_handles = []
    for strange,data in alldata.iteritems():

        fit_params = interpolate(data, funct=options.funct)
        logging.info("fit parameters for mean are {}".format(np.mean(fit_params, axis=0)))

        fitlines = []
        for i in fit_params:
            fitlines.append(gen_line_func(i))

        med_line = gen_line_func(np.mean(fit_params, axis=0))

        logging.info("Using physical point {} to set the scale".format(options.physical))

        intersects = [i[1] for i in fit_params]
        at_zero = np.mean(fit_params, axis=0)[1]


        # hbar*c / 1 fm  = 197.327 MeV
        hbar_c = 197.327

        if options.physical:
            ainvs = [hbar_c * i / options.physical for i in intersects]
        else:
            ainvs = [None for i in intersects]


        legend_handles.append(plot_fitline(data, fitlines, med_line, intersects, ainvs,
                                           ftype, beta, strange, options.output_stub))

        write_data(intersects , ainvs , options.output_stub, "_{}.a_inv".format(strange))

    finish_plot(beta, ftype, legend_handles, options.output_stub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to interpolate the heavy mass")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument("--spinaverage", action="store_true",
                        help="spinaverage vector with pseudoscalar")
    parser.add_argument("-p", "--physical", type=float,
                        help="add physical point")
    parser.add_argument("-m", "--mcc", type=float,
                        help="scale data using given heavy mass")
    parser.add_argument("-f", "--funct", type=str, default="line",
                        help="which function to fit to")
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                        help='files to plot')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    interpolate_wilsonflow(args)

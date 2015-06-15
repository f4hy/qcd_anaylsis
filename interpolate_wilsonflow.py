#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from iminuit import Minuit

from residualmasses import residual_mass

from ensamble_info import data_params, read_fit_mass

colors = ['b', 'k', 'c', 'y', 'm', 'b']

# hbar*c / 1 fm  = 197.327 MeV
hbar_c = 197.327


def read_files(files, xtype, fitdata):
    data = {}

    for f in files:
        logging.info("reading file {}".format(f))
        dp = data_params(f)
        df = pd.read_csv(f, comment='#', names=["flow"])
        xvalue = xvalues(xtype, dp, fitdata, t_0=np.mean(df.flow.values))

        logging.info("read {} giving x={}, y={}".format(dp, np.mean(xvalue), df.flow.mean()))
        data[dp] = (np.mean(xvalue), df.flow.values)

    return data


def xvalues(xaxis_type, data_properties, fitdata, t_0=None):
    logging.info("using xaxis type {}".format(xaxis_type))

    if xaxis_type == "mud":
        residual = residual_mass(data_properties.ud_mass, data_properties.s_mass)
        return (t_0)*pd.Series((data_properties.ud_mass + residual))

    if xaxis_type == "mud_s":
        residual = residual_mass(data_properties.ud_mass, data_properties.s_mass)
        return pd.Serites((data_properties.ud_mass + residual + data_properties.s_mass + residual))

    if xaxis_type == "tmpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        return ((t_0)*pionmass)**2

    if xaxis_type == "t_2mksqr-mpisqr":
        pionmass = read_fit_mass(data_properties, "ud-ud", fitdata)
        kaonmass = read_fit_mass(data_properties, "ud-s", fitdata)
        return (t_0)**2*(2.0*(kaonmass)**2 + (pionmass)**2)


def interpolate(data):

    logging.info("Fitting data")

    dps = []
    xvalues = []
    flows = []
    variances = []

    for dp, d in data.iteritems():
        dps.append(dp)
        xvalues.append(d[0])
        flows.append(np.mean(d[1]))
        n = float(len(d[1]))
        variances.append(((n-1)*np.var(d[1])))

    xvalues = np.array(xvalues)
    flows = np.array(flows)
    variances = np.array(variances)

    def weighted_sqr_diff(C, A):
        sqr_diff = (flows - A*(1+C*xvalues))**2
        return np.sum(sqr_diff/variances)

    dof = float(len(flows)-2)

    guess_A = np.mean(flows)
    guess_slope = (min(flows)-max(flows)) / (max(xvalues) - min(xvalues))

    m = Minuit(weighted_sqr_diff, C=guess_slope, error_C=guess_slope*0.01,
               A=guess_A, error_A=guess_A*0.01, errordef=dof,
               print_level=0, pedantic=True)

    results = m.migrad()
    logging.debug(results)
    logging.debug("variances {}".format(variances))
    logging.debug("std {}".format(np.sqrt(variances)))

    logging.info("chi^2={}, dof={}, chi^2/dof={}".format(m.fval, dof, m.fval/dof))
    logging.info('covariance {}'.format(m.covariance))

    return m


def plot_fitline(data, fit_params, ftype, phys_x, outstub):
    logging.info("ploting")

    c = colors.pop()
    plotsettings = dict(linestyle="none", ms=8, elinewidth=3, capsize=8,
                        capthick=2, mew=3, aa=True, fmt='o')

    xvalues = []

    for dp, d in data.iteritems():
        xvalue, flow = d
        N = len(flow)
        y = np.mean(flow)
        err = np.sqrt((N-1)*(np.std(flow)**2))
        logging.info("{}, {}={}, err={}".format(dp, ftype, y, err))
        # patch = plt.errorbar(xvalue, y, yerr=err, color=c, ecolor=c, mec=c, **plotsettings)

        xvalues.append(xvalue)

    xdata = np.arange(phys_x-0.01, max(xvalues)+0.005, 0.001)
    mydata = fit_params.values["A"]*(1+fit_params.values["C"]*xdata)

    t0_ch = fit_params.values["A"]*(1+fit_params.values["C"]*phys_x)

    plt.plot(xdata, mydata, color=c)

    m = fit_params
    t1 = (m.errors["A"]*(1+m.values["C"]*xdata))**2
    t2 = ((m.values["A"])*xdata*m.errors["C"])**2
    t3 = 2*xdata*mydata*m.covariance[("A", "C")]
    perry = t1+t2+t3

    plt.fill_between(xdata, mydata, mydata+np.sqrt(perry), facecolor=c, alpha=0.1, lw=0, zorder=-10)
    plt.fill_between(xdata, mydata, mydata-np.sqrt(perry), facecolor=c, alpha=0.1, lw=0, zorder=-10)

    t0_ch_variance = (m.errors["A"]*(1+m.values["C"]*phys_x))**2
    t0_ch_variance += ((m.values["A"])*phys_x*m.errors["C"])**2
    t0_ch_variance += 2*phys_x*t0_ch*m.covariance[("A", "C")]

    plt.errorbar(phys_x, t0_ch, yerr=np.sqrt(t0_ch_variance),
                 color="r", mec="r", **plotsettings)

    t0_ch_std = np.sqrt(t0_ch_variance)

    logging.info("Determined at physical point t_0 = {:.5f} +/- {:.5f}".format(t0_ch, t0_ch_std))

    return t0_ch, t0_ch_std


def finish_plot(beta, ftype, xlabel, legend_handles, outstub):
    fontsettings = dict(fontsize=20)

    plt.title(r'$\beta={}$    ${}$'.format(beta, ftype), **fontsettings)
    plt.ylabel("${} / a$".format(ftype), **fontsettings)
    plt.xlabel(xlabel, **fontsettings)

    plt.legend(handles=sorted(legend_handles), loc=0, **fontsettings)

    if outstub is not None:
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        filename = outstub+".png"
        logging.info("Saving plot to {}".format(filename))
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


def write_data(t0_ch, a_inv, a_inv_err, output_stub, suffix):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing a_inv to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        ofile.write("{}, {} +/- {}\n".format(t0_ch, a_inv, a_inv_err))


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

    alldata = read_files(options.files, options.xaxis, options.fitdata)

    if options.xaxis == "mud":
        phys_x = 0
    if options.xaxis == "tmpisqr":
        phys_x = ((0.1465/hbar_c)*135.0)**2
    if options.xaxis == "t_2mksqr-mpisqr":
        phys_x = ((0.1465/hbar_c)**2)*(2*(495**2) + 138.0**2)

    logging.info("Attempting to extrapolate to {}={}".format(options.xaxis, phys_x))

    legend_handles = []

    stranges = set(i.s_mass for i in alldata.keys())

    if options.seperate_strange:
        logging.info("fitting strangeses seperately")
        groups = [{k: v for k, v in alldata.iteritems() if k.s_mass == s} for s in stranges]
    else:
        logging.info("fitting all data together")
        groups = [alldata]

    for data_group in groups:

        fit_params = interpolate(data_group)

        fitstring = ", ".join(["{}: {:.6f} +/- {:.6f}".format(k, v, fit_params.errors[k]) for
                               k, v in fit_params.values.iteritems()])
        logging.info("fit parameters found to be {}".format(fitstring))

        phys_t0 = 0.1465

        t0_ch, phys_err = plot_fitline(data_group, fit_params, ftype, phys_x,
                                        options.output_stub)

        ainv = hbar_c * t0_ch / phys_t0
        ainv_err = hbar_c * phys_err / phys_t0
        logging.info("ainv = {} +/- {}".format(ainv, ainv_err))

        write_data(t0_ch, ainv, ainv_err, options.output_stub, "_a_inv")

    xlabels = {"mud": r"$t_0^{1/2} m_{ud}$", "tmpisqr": r"$t_0 (m_{\pi})^2$",
               "t_2mksqr-mpisqr": r"$t_0 (2m_k^2+m_{\pi}^2)^2$"}
    finish_plot(beta, ftype, xlabels[options.xaxis], legend_handles, options.output_stub)


if __name__ == "__main__":

    axis_choices = ["mud", "mud_s", "mpi", "tmpisqr", "t_2mksqr-mpisqr"]

    parser = argparse.ArgumentParser(description="script to interpolate the heavy mass")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--seperate_strange", action="store_true",
                        help="fit different strange values seperately")
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
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument("--xaxis", required=False, choices=axis_choices,
                        help="what to set on the xaxis", default="mud")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    if args.xaxis == "tmpisqr":
        for f in args.files:
            if "t0" not in f:
                raise argparse.ArgumentTypeError("tmpisqr requires t0 data".format(f))

    interpolate_wilsonflow(args)
